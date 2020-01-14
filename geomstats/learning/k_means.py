import random

import geomstats.backend as gs
from geomstats.learning._template import TemplateTransformer


class K_Means(TemplateTransformer):

    def __init__(self, n_clusters, metric, init="random",
                 n_init=1, n_jobs=None, tol=1e-2):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : If given only give the metric for each labels

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.metric = metric
        self.n_jobs = n_jobs
        self.n_clusters = n_clusters
        self.tol = tol

    def fit(self, X, Y=None, max_iter=100, eps=1e-4, convergence_value=1e-2):
        """Predict for each data point the closest center in terms of metric distance

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : If given only give the metric for each labels

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        belongs = gs.zeros(X.shape[0])
        self.centroids = gs.vstack([gs[random.randint(0, self.n_clusters-1)]
                                    for i in range(self.n_clusters)])
        index = 0
        while(index < max_iter):
            index += 1
            # expectation
            dists = gs.vstack([self.metric.dist(self.centroids[i], X)
                              for i in range(self.n_clusters)])
            belongs = gs.argmin(dists, -1)
            # maximisation
            old_centroids = self.centroids
            self.centroids = gs.vstack([self.metric.mean(
                                        gs.squeeze(X[belongs == i]))
                                        for i in range(self.n_clusters)])
            # test convergence
            '''
            Maybe Change it later
            '''
            if(gs.mean(self.metric.dist(old_centroids, self.centroids))
               < self.tol):
                # convergence reached
                return gs.copy(self.centroids)

    def predict(self, X):
        # finding closest mean
        dists = gs.vstack([self.metric.dist(self.centroids[i], X)
                           for i in range(self.n_clusters)])
        belongs = gs.argmin(dists, -1)
        return belongs

    def fit_predict(self, X, Y):
        self.fit(X, Y)
        return self.predict(X)
