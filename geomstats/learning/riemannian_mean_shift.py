"""
Riemannian mean-shift clustering
Lead author: Nina Miolane and Shubham Talbar
"""

import joblib
from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean


class RiemannianMeanShift(ClusterMixin, BaseEstimator):
    """
    Class for Riemannian Mean Shift algorithm on manifolds.
    Mean Shift is a procedure for locating the maxima - the modes of a
    density function given discrete data sampled from that function. It is
    an iterative method for finding the centers of a collection of clusters.

    Following implementation assumes a flat kernel method.

    Parameters
    ----------
    manifold : object of class RiemannianManifold
        Geomstats Riemannian manifold on which the
        Riemannian Mean Shift algorithm is to applied.
    metric : object of class RiemannianMetric
        Geomstats Riemannian metric associated to the space used.
    bandwidth : size of neighbourhood around each center
        All points in 'bandwidth' size around center are considered for
        calculating new mean centers.
    tot : float, stopping condition
        Computation of subsequent mean centers is stopped when the distance
        between them is less than 'tot'
    mean : Mean method used to calculate the new center of a cluster
    n_centers : Total initial centers
    n_jobs : Number of parallel threads to be initiated for parallel computation
    max_iter : int
        Upper bound on total number of iterations for the centers to converge
    init_centers : Initializing centers, either from the given input points or
             random points uniformly distributed in the input manifold
    kernel : Weighing function to assign kernel weights to each center
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

        print(type(max_iter))
        print(max_iter)

    def __intersets_distances(self, points_A, points_B, **joblib_kwargs):
        """Parallel computation of distances between two sets of points"""
        n_A, n_B = points_A.shape[0], points_B.shape[0]

        @joblib.delayed
        @joblib.wrap_non_picklable_objects
        def pickable_dist(x, y):
            """Riemannian metric between points x & y"""
            return self.metric.dist(x, y)

        pool = joblib.Parallel(n_jobs=self.n_jobs, **joblib_kwargs)
        out = pool(
            pickable_dist(points_A[i, :], points_B[j, :])
            for i in range(n_A)
            for j in range(n_B)
        )

        return gs.array(out).reshape((n_A, n_B))

    def fit(self, X):
        """Fit centers in all the input points to find 'n_centers' number of clusters"""

        @joblib.delayed
        @joblib.wrap_non_picklable_objects
        def pickable_mean(points, weights):
            """Frechet Mean of all points weighted by weights"""
            return self.mean.fit(points, weights=weights).estimate_

        # 'from_points' is the default initialization for centers used
        if self.init_centers == "from_points":
            n_points = X.shape[0]
            centers = X[gs.random.randint(n_points, size=self.n_centers), :]
        if self.init_centers == "random_uniform":
            n_centers = centers.shape[0]
            centers = self.manifold.random_uniform(n_samples=n_centers)

        print(type(self.max_iter))
        print(self.max_iter)
        for _ in range(self.max_iter):
            dists = self.__intersets_distances(centers, X)

            # assuming the use of 'flat' kernel
            if self.kernel == "flat":
                weights = gs.ones_like(dists)

            weights[dists > self.bandwidth] = 0.0
            weights = weights / gs.sum(weights, axis=1, keepdims=1)

            points_to_average, nonzero_weights = [], []

            for j in range(self.n_centers):
                indexes = gs.where(weights[j] > 0)
                nonzero_weights += [
                    weights[j][indexes],
                ]
                points_to_average += [
                    X[indexes],
                ]

            # compute Frechet means in parallel
            pool = joblib.Parallel(n_jobs=self.n_jobs)
            out = pool(
                pickable_mean(points_to_average[j], nonzero_weights[j])
                for j in range(self.n_centers)
            )

            new_centers = gs.array(out)

            displacements = [self.metric.dist(centers[j], new_centers[j])]
            centers = new_centers

            if (gs.array(displacements) < self.tol).all():
                break

        self.centers = centers

    def predict(self, points):
        """Predict the closest cluster each point in 'points' belongs to"""
        if self.centers is None:
            raise Exception("Not fitted")

        out = []
        for i in range(points.shape[0]):
            j_closest_center = self.metric.closest_neighbor_index(
                points[i, :], self.centers
            )
            out.append(self.centers[j_closest_center, :])

        return gs.array(out)
