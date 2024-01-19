"""K-means clustering.

Lead author: Hadi Zaatiti.
"""

import logging
from random import randint

from scipy.stats import rv_discrete
from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning._template import TransformerMixin
from geomstats.learning.frechet_mean import FrechetMean, LinearMean


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
        How to initialize cluster centers at the beginning of the algorithm. The
        choice 'random' will select training points as initial cluster centers
        uniformly at random. The choice 'kmeans++' selects cluster centers
        heuristically to improve the convergence rate. When providing an array
        of shape ``(n_clusters, n_features)``, the cluster centers are chosen as the
        rows of that array. When providing a callable, it receives as arguments
        the argument ``X`` to :meth:`fit` and the number of cluster centers
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

        self.init_cluster_centers_ = None

        if isinstance(space, Euclidean):
            self.mean_estimator = LinearMean(space=space)
        else:
            self.mean_estimator = FrechetMean(
                space=space,
                method="default",
            ).set(max_iter=100, init_step_size=1.0)

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _pick_init_cluster_centers(self, X):
        n_samples = X.shape[0]

        if isinstance(self.init, str):
            if self.init == "kmeans++":
                cluster_centers = [X[randint(0, n_samples - 1)]]
                for i in range(self.n_clusters - 1):
                    dists = gs.array(
                        [
                            self.space.metric.dist(cluster_centers[j], X)
                            for j in range(i + 1)
                        ]
                    )
                    dists_to_closest_cluster_center = gs.amin(dists, axis=0)
                    indices = gs.arange(n_samples)
                    weights = dists_to_closest_cluster_center / gs.sum(
                        dists_to_closest_cluster_center
                    )
                    index = rv_discrete(
                        values=(gs.to_numpy(indices), gs.to_numpy(weights))
                    ).rvs()
                    cluster_centers.append(X[index])
            elif self.init == "random":
                cluster_centers = [
                    X[randint(0, n_samples - 1)] for i in range(self.n_clusters)
                ]
            else:
                raise ValueError(
                    f"Unknown initial cluster centers method '{self.init}'."
                )

            cluster_centers = gs.stack(cluster_centers, axis=0)
        else:
            if callable(self.init):
                cluster_centers = self.init(X, self.n_clusters)
            else:
                cluster_centers = self.init

            if cluster_centers.shape[0] != self.n_clusters:
                raise ValueError("Need as many initial cluster centers as clusters.")

            if cluster_centers.shape[1] != X.shape[1]:
                raise ValueError(
                    "Dimensions of initial cluster centers and "
                    "training data do not match."
                )

        return cluster_centers

    def fit(self, X):
        """Provide cluster centers and data labels.

        Alternate between computing the mean of each cluster
        and labelling data according to the new positions of the cluster centers.

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

        cluster_centers = self._pick_init_cluster_centers(X)
        self.init_cluster_centers_ = gs.copy(cluster_centers)

        dists = [
            gs.to_ndarray(self.space.metric.dist(cluster_centers[i], X), 2, 1)
            for i in range(self.n_clusters)
        ]
        dists = gs.hstack(dists)
        self.labels_ = gs.argmin(dists, 1)

        for index in range(self.max_iter):
            if self.verbose > 0:
                logging.info(f"Iteration {index}...")

            old_cluster_centers = gs.copy(cluster_centers)
            for i in range(self.n_clusters):
                fold = X[self.labels_ == i]

                if len(fold) > 0:
                    self.mean_estimator.fit(fold)
                    cluster_centers[i] = self.mean_estimator.estimate_
                else:
                    cluster_centers[i] = X[randint(0, n_samples - 1)]

            dists = [
                gs.to_ndarray(self.space.metric.dist(cluster_centers[i], X), 2, 1)
                for i in range(self.n_clusters)
            ]
            dists = gs.hstack(dists)
            self.labels_ = gs.argmin(dists, 1)
            dists_to_closest_cluster_center = gs.amin(dists, 1)
            self.inertia_ = gs.sum(dists_to_closest_cluster_center**2)
            cluster_centers_distances = self.space.metric.dist(
                old_cluster_centers, cluster_centers
            )
            if self.verbose > 0:
                logging.info(
                    f"Convergence criterion at the end of iteration {index} "
                    f"is {gs.mean(cluster_centers_distances)}."
                )

            if gs.mean(cluster_centers_distances) < self.tol:
                if self.verbose > 0:
                    logging.info(f"Convergence reached after {index} iterations.")

                break
        else:
            logging.warning(
                f"K-means maximum number of iterations {self.max_iter} reached. "
                "The mean may be inaccurate."
            )

        self.cluster_centers_ = cluster_centers

        return self

    def predict(self, X):
        """Predict the labels for each data point.

        Label each data point with the cluster having the nearest
        cluster center using metric distance.

        Parameters
        ----------
        X : array-like, shape[n_samples, n_features]
            Input data.

        Returns
        -------
        labels : array-like, shape=[n_samples,]
            Array of predicted cluster indices for each sample.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("fit needs to be called first.")
        dists = gs.stack(
            [
                self.space.metric.dist(cluster_center, X)
                for cluster_center in self.cluster_centers_
            ],
            axis=1,
        )
        dists = gs.squeeze(dists)

        labels = gs.argmin(dists, -1)

        return labels
