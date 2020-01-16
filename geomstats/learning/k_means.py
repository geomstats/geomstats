from random import randint

from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.learning._template import TransformerMixin


class RiemannianKMeans(TransformerMixin, ClusterMixin, BaseEstimator):

    def __init__(self, riemannian_metric, n_clusters=8, init='random',
                 tol=1e-2, verbose=0):
        """ K-Means algorithm using Riemannian manifolds

        Parameters
        ----------
        n_clusters : Number of clusters (k value of the k-means)

        riemannian_metric : The geomstats riemmanian metric associate to
                            the space used

        init : How to init centroids at the beginning of the algorithm.
               'random' : will select random uniformally train point as
                         initial centroids

        tol : convergence factor. If the difference of mean distance
             between two step is lower than tol

        verbose : if verbose > 0, information will be print during learning

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.riemannian_metric = riemannian_metric
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, max_iter=100):
        """Predict for each data point the closest center in terms of
            riemannian_metric distance

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        max_iter : Maximum number of iterations

        Returns
        -------
        self : object
            Return centroids array
        """
        n_samples = X.shape[0]
        belongs = gs.zeros(n_samples)
        self.centroids = [gs.expand_dims(X[randint(0, n_samples-1)], 0)
                          for i in range(self.n_clusters)]
        self.centroids = gs.concatenate(self.centroids)
        print(self.centroids)
        index = 0
        while index < max_iter:
            index += 1

            dists = [self.riemannian_metric.dist(self.centroids[i], X)
                     for i in range(self.n_clusters)]
            dists = gs.hstack(dists)
            belongs = gs.argmin(dists, -1)
            old_centroids = gs.copy(self.centroids)
            for i in range(self.n_clusters):
                fold = gs.squeeze(X[belongs == i])
                if len(fold) > 0:
                    print(fold.shape)
                    self.centroids[i] = self.riemannian_metric.mean(fold)

                else:
                    self.centroids[i] = X[randint(0, n_samples-1)]

            centroids_distances = self.riemannian_metric.dist(old_centroids,
                                                              self.centroids)

            if gs.mean(centroids_distances) < self.tol:
                if self.verbose > 0:
                    print("Convergence Reached after ", index, " iterations")

                return gs.copy(self.centroids)

        return gs.copy(self.centroids)

    def predict(self, X):

        """Predict for each data point the closest center in terms of
            riemannian_metric distance

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Return array containing for each point the cluster associated
        """
        dists = gs.hstack([self.riemannian_metric.dist(self.centroids[i], X)
                           for i in range(self.n_clusters)])
        belongs = gs.argmin(dists, -1)
        return belongs
