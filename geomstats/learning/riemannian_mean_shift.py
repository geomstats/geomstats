"""Riemannian mean-shift clustering.

Lead author: Nina Miolane and Shubham Talbar.
"""

import joblib
from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean


class RiemannianMeanShift(ClusterMixin, BaseEstimator):
    """Class for Riemannian Mean Shift algorithm on manifolds.

    Mean Shift is a procedure for locating the maxima - the modes of a
    density function given discrete data sampled from that function. It is
    an iterative method for finding the centers of a collection of clusters.

    Following implementation assumes a flat kernel method.

    Parameters
    ----------
    manifold : object of class RiemannianManifold
        Geomstats Riemannian manifold on which the
        Riemannian Mean Shift algorithm is to be applied.
    metric : object of class RiemannianMetric
        Geomstats Riemannian metric associated to the space used.
    bandwidth : float
        Size of neighbourhood around each center. All points in 'bandwidth'
        size around center are considered for calculating new mean centers.
    tol : float
        Stopping condition. Computation of subsequent mean centers is stopped
        when the distance between them is less than 'tol'.
        Optional, default : 1e-2.
    n_centers : int
        Number of centers.
        Optional, default : 1.
    n_jobs : int
        Number of parallel threads to be initiated for parallel jobs.
        Optional, default : 1.
    max_iter : int
        Upper bound on total number of iterations for the centers to converge.
        Optional, default : 100.
    init_centers : str
        Initializing centers, either from the given input points or
        random points uniformly distributed in the input manifold.
        Optional, default : "from_points".
    kernel : str
        Weighing function to assign kernel weights to each center.
        Optional, default : "flat".
    """

    def __init__(
        self,
        manifold,
        metric,
        bandwidth,
        tol=1e-2,
        n_centers=1,
        n_jobs=1,
        max_iter=100,
        init_centers="from_points",
        kernel="flat",
        **frechet_mean_kwargs
    ):
        self.manifold = manifold
        self.metric = metric
        self.bandwidth = bandwidth
        self.tol = tol
        self.mean = FrechetMean(self.metric, **frechet_mean_kwargs)
        self.n_centers = n_centers
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.init_centers = init_centers
        self.kernel = kernel
        self.centers = None

    def dist_intersets(self, points_a, points_b, **joblib_kwargs):
        """Parallel computation of distances between two sets of points.

        Parameters
        ----------
        points_a : array-like, shape=[..., n_features]
            Clusters of points.
        points_b : array-like, shape=[..., n_features]
            Clusters of points.
        """
        n_a, n_b = points_a.shape[0], points_b.shape[0]

        @joblib.delayed
        @joblib.wrap_non_picklable_objects
        def pickable_dist(x, y):
            """Riemannian distance between points x & y.

            Parameters
            ----------
            x : single point, shape=[1, n_features]
                Single point on manifold.
            y : array-like, shape=[1, n_features]
                Single point on manifold.
            """
            return self.metric.dist(x, y)

        pool = joblib.Parallel(n_jobs=self.n_jobs, **joblib_kwargs)
        out = pool(
            pickable_dist(point_a, point_b)
            for point_a in points_a
            for point_b in points_b
        )

        return gs.array(out).reshape((n_a, n_b))

    def fit(self, x, y=None):
        """Fit centers in all the input points.

        Parameters
        ----------
        x : array-like, shape=[..., n_features]
            Clusters of points.
        """

        @joblib.delayed
        @joblib.wrap_non_picklable_objects
        def pickable_mean(points, weights):
            """Frechet Mean of all points weighted by weights.

            Parameters
            ----------
            points : array-like, shape=[..., n_features]
                Clusters of points.
            weights : array-like,
                Weight associated with each point in cluster.
            """
            return self.mean.fit(points, weights=weights).estimate_

        if self.init_centers == "from_points":
            n_points = x.shape[0]
            centers = x[gs.random.randint(n_points, size=(self.n_centers,)), :]
        elif self.init_centers == "random_uniform":
            centers = self.manifold.random_uniform(n_samples=self.n_centers)

        for _ in range(self.max_iter):
            dists = self.dist_intersets(centers, x)

            if self.kernel == "flat":
                weights = gs.ones_like(dists)

            weights[dists > self.bandwidth] = 0.0
            weights = weights / gs.sum(weights, axis=1, keepdims=True)

            points_to_average, nonzero_weights = [], []

            for j in range(self.n_centers):
                indexes = gs.where(weights[j] > 0)
                nonzero_weights += [
                    weights[j][indexes],
                ]
                points_to_average += [
                    x[indexes],
                ]

            pool = joblib.Parallel(n_jobs=self.n_jobs)
            out = pool(
                pickable_mean(points_to_average[j], nonzero_weights[j])
                for j in range(self.n_centers)
            )

            new_centers = gs.array(out)

            displacements = [self.metric.dist(centers, new_centers)]
            centers = new_centers

            if (gs.array(displacements) < self.tol).all():
                break

        self.centers = centers

    def predict(self, points):
        """Predict the closest cluster each point in 'points' belongs to.

        Parameters
        ----------
        points : array-like, shape=[..., n_features]
            Clusters of points.
        """
        if self.centers is None:
            raise Exception("Not fitted")

        closest_centers = []
        for point in points:
            closest_center = self.metric.closest_neighbor_index(point, self.centers)
            closest_centers.append(self.centers[closest_center, :])

        return gs.array(closest_centers)
