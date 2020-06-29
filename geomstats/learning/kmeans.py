"""K-means clustering."""

import logging
from random import randint

from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.learning._template import TransformerMixin
from geomstats.learning.frechet_mean import FrechetMean


class RiemannianKMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """Class for k-means clustering on manifolds.

    K-means algorithm using Riemannian manifolds.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (k value of the k-means).
        Optional, default: 8.
    metric : object of class RiemannianMetric
        The geomstats Riemmanian metric associate to the space used.
    init : str
        How to initialize centroids at the beginning of the algorithm. The
        choice 'random' will select training points as initial centroids
        uniformly at random.
        Optional, default: 'random'.
    tol : float
        Convergence factor. Convergence is achieved when the difference of mean
        distance between two steps is lower than tol.
        Optional, default: 1e-2.
    verbose : int
        If verbose > 0, information will be printed during learning.
        Optional, default: 0.

    Example
    -------
    Available example on the Poincaré Ball and Hypersphere manifolds
    :mod:`examples.plot_kmeans_manifolds`
    """

    def __init__(
            self, metric, n_clusters=8, init='random',
            tol=1e-2, mean_method='default', verbose=0, point_type='vector'):
        self.n_clusters = n_clusters
        self.init = init
        self.metric = metric
        self.tol = tol
        self.verbose = verbose
        self.mean_method = mean_method
        self.point_type = point_type

        self.centroids = None

    def fit(self, X, max_iter=100):
        """Provide clusters centroids and data labels.

        Alternate between computing the mean of each cluster
        and labelling data according to the new positions of the centroids.

        Parameters
        ----------
        X : array-like, shape=[..., n_features]
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        max_iter : int
            Maximum number of iterations.
            Optional, default: 100.

        Returns
        -------
        self : array-like, shape=[n_clusters,]
            Centroids.
        """
        n_samples = X.shape[0]
        self.centroids = [gs.expand_dims(X[randint(0, n_samples - 1)], 0)
                          for i in range(self.n_clusters)]
        self.centroids = gs.concatenate(self.centroids, axis=0)
        index = 0
        while index < max_iter:
            index += 1

            dists = [gs.to_ndarray(
                     self.metric.dist(self.centroids[i], X), 2, 1)
                     for i in range(self.n_clusters)]
            dists = gs.hstack(dists)
            belongs = gs.argmin(dists, 1)
            old_centroids = gs.copy(self.centroids)
            for i in range(self.n_clusters):
                fold = gs.squeeze(X[belongs == i])

                if len(fold) > 0:

                    mean = FrechetMean(
                        metric=self.metric,
                        method=self.mean_method,
                        max_iter=150,
                        point_type=self.point_type)
                    mean.fit(fold)

                    self.centroids[i] = mean.estimate_
                else:
                    self.centroids[i] = X[randint(0, n_samples - 1)]

            centroids_distances = self.metric.dist(
                old_centroids, self.centroids)

            if gs.mean(centroids_distances) < self.tol:
                if self.verbose > 0:
                    logging.info('Convergence reached after {} '
                                 'iterations'.format(index))

                if self.n_clusters == 1:
                    self.centroids = gs.squeeze(self.centroids, axis=0)

                return gs.copy(self.centroids)

        if index == max_iter:
            logging.warning('K-means maximum number of iterations {} reached. '
                            'The mean may be inaccurate'.format(max_iter))

        if self.n_clusters == 1:
            self.centroids = gs.squeeze(self.centroids, axis=0)
        return gs.copy(self.centroids)

    def predict(self, X):
        """Predict the labels for each data point.

        Label each data point with the cluster having the nearest
        centroid using metric distance.

        Parameters
        ----------
        X : array-like, shape=[..., n_features]
            Input data.

        Returns
        -------
        self : array-like, shape=[...,]
            Array of predicted cluster indices for each sample.
        """
        if self.centroids is None:
            raise RuntimeError('fit needs to be called first.')
        dists = gs.stack(
            [self.metric.dist(centroid, X)
             for centroid in self.centroids],
            axis=1)
        dists = gs.squeeze(dists)

        belongs = gs.argmin(dists, -1)

        return belongs
