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

    riemannian_metric : object of class RiemannianMetric
        The geomstats Riemmanian metric associate to the space used.

    init : str
        How to initialize centroids at the beginning of the algorithm. The
        choice 'random' will select training points as initial centroids
        uniformly at random.

    tol : float
        Convergence factor. Convergence is achieved when the difference of mean
        distance between two steps is lower than tol.

    verbose : int
        If verbose > 0, information will be printed during learning.

    Example
    -------
    Available example on the Poincaré Ball and Hypersphere manifolds
    :mod:`examples.plot_kmeans_manifolds`
    """

    def __init__(self, riemannian_metric, n_clusters=8, init='random',
                 tol=1e-2, mean_method='default', verbose=0):
        self.n_clusters = n_clusters
        self.init = init
        self.riemannian_metric = riemannian_metric
        self.tol = tol
        self.verbose = verbose
        self.mean_method = mean_method

        self.centroid = None

    def fit(self, X, max_iter=100):
        """Provide clusters centroids and data labels.

        Alternate between computing the mean of each cluster
        and labelling data according to the new positions of the centroids.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        max_iter : int
            Maximum number of iterations

        Returns
        -------
        self : array-like, shape=[n_clusters,]
            centroids array
        """
        n_samples = X.shape[0]
        belongs = gs.zeros(n_samples)
        self.centroids = [gs.expand_dims(X[randint(0, n_samples - 1)], 0)
                          for i in range(self.n_clusters)]
        self.centroids = gs.concatenate(self.centroids)
        index = 0
        while index < max_iter:
            index += 1

            dists = [gs.to_ndarray(
                     self.riemannian_metric.dist(self.centroids[i], X), 2, 1)
                     for i in range(self.n_clusters)]
            dists = gs.hstack(dists)
            belongs = gs.argmin(dists, 1)
            old_centroids = gs.copy(self.centroids)
            for i in range(self.n_clusters):
                fold = gs.squeeze(X[belongs == i])

                if len(fold) > 0:

                    mean = FrechetMean(
                        metric=self.riemannian_metric,
                        method=self.mean_method,
                        max_iter=150)
                    mean.fit(fold)

                    self.centroids[i] = mean.estimate_
                else:
                    self.centroids[i] = X[randint(0, n_samples - 1)]

            centroids_distances = self.riemannian_metric.dist(old_centroids,
                                                              self.centroids)

            if gs.mean(centroids_distances) < self.tol:
                if self.verbose > 0:
                    logging.info('Convergence Reached after {} '
                                 'iterations'.format(index))

                return gs.copy(self.centroids)

        if index == max_iter:
            logging.warning('K-means maximum number of iterations {} reached.'
                            'The mean may be inaccurate'.format(max_iter))

        return gs.copy(self.centroids)

    def predict(self, X):
        """Predict the labels for each data point.

        Label each data point with the cluster having the nearest
        centroid using riemannian_metric distance.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Input data

        Returns
        -------
        self : array-like, shape=[n_samples,]
            Array of predicted cluster indices for each sample
        """
        assert self.centroids is not None, 'fit needs to be called first'
        dists = gs.hstack([self.riemannian_metric.dist(centroid, X)
                           for centroid in self.centroids])
        belongs = gs.argmin(dists, -1)
        return belongs
