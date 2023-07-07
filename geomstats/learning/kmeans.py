"""K-means clustering.

Lead author: Hadi Zaatiti.
"""

import logging
from random import randint

from scipy.stats import rv_discrete
from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.learning._template import TransformerMixin
from geomstats.learning.frechet_mean import FrechetMean


class RiemannianKMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """Class for k-means clustering on manifolds.

    K-means algorithm using Riemannian manifolds.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    n_clusters : int
        Number of clusters (k value of the k-means).
        Optional, default: 8.
    init : str or callable or array-like, shape=[n_clusters, n_features]
        How to initialize centroids at the beginning of the algorithm. The
        choice 'random' will select training points as initial centroids
        uniformly at random. The choice 'kmeans++' selects centroids
        heuristically to improve the convergence rate. When providing an array
        of shape ``(n_clusters, n_features)``, the centroids are chosen as the
        rows of that array. When providing a callable, it receives as arguments
        the argument ``X`` to :meth:`fit` and the number of centroids
        ``n_clusters`` and is expected to return an array as above.
        Optional, default: 'random'.
    tol : float
        Convergence factor. Convergence is achieved when the difference of mean
        distance between two steps is lower than tol.
        Optional, default: 1e-2.
    max_iter : int
        Maximum number of iterations.
        Optional, default: 100
    verbose : int
        If verbose > 0, information will be printed during learning.
        Optional, default: 0.

    Notes
    -----
    * Required metric methods: `dist`.

    Example
    -------
    Available example on the Poincaré Ball and Hypersphere manifolds
    :mod:`examples.plot_kmeans_manifolds`
    """

    def __init__(
        self,
        space,
        n_clusters=8,
        init="random",
        tol=1e-2,
        max_iter=100,
        verbose=0,
    ):
        self.space = space

        self.n_clusters = n_clusters
        self.init = init
        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter

        self.init_centroids = None

        self.mean_estimator = FrechetMean(
            space=space,
            method="default",
        ).set(max_iter=100, init_step_size=1.0)

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def _pick_init_centroids(self, X):
        # TODO: clean this part of the code?
        n_samples = X.shape[0]

        if isinstance(self.init, str):
            if self.init == "kmeans++":
                centroids = [X[randint(0, n_samples - 1)]]
                for i in range(self.n_clusters - 1):
                    dists = [
                        self.space.metric.dist(centroids[j], X) for j in range(i + 1)
                    ]
                    dists_to_closest_centroid = gs.amin(dists, axis=0)
                    indices = gs.arange(n_samples)
                    weights = dists_to_closest_centroid / gs.sum(
                        dists_to_closest_centroid
                    )
                    index = rv_discrete(values=(indices, weights)).rvs()
                    centroids.append(X[index])
            elif self.init == "random":
                centroids = [
                    X[randint(0, n_samples - 1)] for i in range(self.n_clusters)
                ]
            else:
                raise ValueError(f"Unknown initial centroids method '{self.init}'.")

            centroids = gs.stack(centroids, axis=0)
        else:
            if callable(self.init):
                centroids = self.init(X, self.n_clusters)
            else:
                centroids = self.init

            if centroids.shape[0] != self.n_clusters:
                raise ValueError("Need as many initial centroids as clusters.")

            if centroids.shape[1] != X.shape[1]:
                raise ValueError(
                    "Dimensions of initial centroids and training data do not match."
                )

        return centroids

    def fit(self, X):
        """Provide clusters centroids and data labels.

        Alternate between computing the mean of each cluster
        and labelling data according to the new positions of the centroids.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object
            Returns self.
        """
        n_samples = X.shape[0]
        if self.verbose > 0:
            logging.info("Initializing...")

        centroids = self._pick_init_centroids(X)
        self.init_centroids = gs.copy(centroids)

        dists = [
            gs.to_ndarray(self.space.metric.dist(centroids[i], X), 2, 1)
            for i in range(self.n_clusters)
        ]
        dists = gs.hstack(dists)
        self.labels_ = gs.argmin(dists, 1)

        for index in range(self.max_iter):
            if self.verbose > 0:
                logging.info(f"Iteration {index}...")

            old_centroids = gs.copy(centroids)
            for i in range(self.n_clusters):
                fold = X[self.labels_ == i]

                if len(fold) > 0:
                    self.mean_estimator.fit(fold)
                    centroids[i] = self.mean_estimator.estimate_
                else:
                    centroids[i] = X[randint(0, n_samples - 1)]

            dists = [
                gs.to_ndarray(self.space.metric.dist(centroids[i], X), 2, 1)
                for i in range(self.n_clusters)
            ]
            dists = gs.hstack(dists)
            self.labels_ = gs.argmin(dists, 1)
            dists_to_closest_centroid = gs.amin(dists, 1)
            self.inertia_ = gs.sum(dists_to_closest_centroid**2)
            centroids_distances = self.space.metric.dist(old_centroids, centroids)
            if self.verbose > 0:
                logging.info(
                    f"Convergence criterion at the end of iteration {index} "
                    f"is {gs.mean(centroids_distances)}."
                )

            if gs.mean(centroids_distances) < self.tol:
                if self.verbose > 0:
                    logging.info(f"Convergence reached after {index} iterations.")

                break
        else:
            logging.warning(
                f"K-means maximum number of iterations {self.max_iter} reached. "
                "The mean may be inaccurate."
            )

        self.centroids_ = centroids

        return self

    def predict(self, X):
        """Predict the labels for each data point.

        Label each data point with the cluster having the nearest
        centroid using metric distance.

        Parameters
        ----------
        X : array-like, shape[n_samples, n_features]
            Input data.

        Returns
        -------
        labels : array-like, shape=[n_samples,]
            Array of predicted cluster indices for each sample.
        """
        if self.centroids_ is None:
            raise RuntimeError("fit needs to be called first.")
        dists = gs.stack(
            [self.space.metric.dist(centroid, X) for centroid in self.centroids_],
            axis=1,
        )
        dists = gs.squeeze(dists)

        labels = gs.argmin(dists, -1)

        return labels
