import random

import geomstats.backend as gs
from geomstats.learning._template import TemplateTransformer


class KMeans(TemplateTransformer):

    def __init__(self, n_clusters, metric, init="random",
                 n_init=1, n_jobs=None, tol=1e-2, verbose=0):
        """ K-Means algorithm using Riemannian manifolds

        Parameters
        ----------
        n_clusters : Number of clusters (k value of the k-means)

        metric : the metric associate to the space used

        init : How to init centroids at the beginning of the algorithm.
            Only radom is implemented yet.

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
        self.n_init = n_init
        self.metric = metric
        self.n_jobs = n_jobs
        self.n_clusters = n_clusters
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, Y=None, max_iter=100):
        """Predict for each data point the closest center in terms of metric distance

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        max_iter : Maximum number of iteration

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        belongs = gs.zeros(X.shape[0])
        self.centroids = gs.concatenate([gs.expand_dims(X[
                                         random.randint(0, X.shape[0]-1)], 0)
                                         for i in range(self.n_clusters)])
        index = 0
        while(index < max_iter):
            index += 1
            # expectation
            dists = gs.hstack([self.metric.dist(self.centroids[i], X)
                              for i in range(self.n_clusters)])
            belongs = gs.argmin(dists, -1)
            # maximisation
            old_centroids = gs.copy(self.centroids)

            for i in range(self.n_clusters):
                fold = gs.squeeze(X[belongs == i])
                if(len(fold) > 0):
                    self.centroids[i] = self.metric.mean(fold)
                else:
                    self.centroids[i] = X[random.randint(0, X.shape[0]-1)]

            # test convergence
            if(gs.mean(self.metric.dist(old_centroids, self.centroids))
               < self.tol):
                print("Convergence Reached after ", index, " iterations")
                # convergence reached
                return gs.copy(self.centroids)

    def predict(self, X):
        # finding closest mean
        dists = gs.hstack([self.metric.dist(self.centroids[i], X)
                           for i in range(self.n_clusters)])
        belongs = gs.argmin(dists, -1)
        return belongs
