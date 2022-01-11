"""Riemannian mean-shift clustering

Lead author: Nina Miolane"""

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from geomstats.learning.frechet_mean import FrechetMean


class RiemannianMeanShift(ClusterMixin, BaseEstimator):
    def __init__(self, manifold, metric, bandwidth, tol=1e-2, **FrechetMean_kwargs):
        """Class for Riemannian Mean Shift algorithm on manifolds.
        Mean Shift is a procedure for locating the maxima - the modes of a
        density function given discrete data sampled from that function. It is
        an iterative method for finding the centers of a collection of clusters.

        Following implementation assumes a numpy backend and a flat kernel method.

        Parameters
        ----------
        manifold : object of class RiemannianManifold
            The Geomstats Riemannian manifold on which the
            Riemannian Mean Shift algorithm is to applied.
        metric : object of class RiemannianMetric
            The Geomstats Riemannian metric associated to the space used.
        bandwidth : size of neighbourhood around each center
            All points in 'bandwidth' size around center are considered for
            calculating new mean centers.
        tot : float, stopping condition
            Computation of subsequent mean centers is stopped when the distance
            between them is less than 'tot'
        mean : mean method used to calculate the new center of a cluster
        """
        self.manifold = manifold
        self.metric = metric
        self.bandwidth = bandwidth
        self.tol = tol
        self.mean = FrechetMean(self.metric, **FrechetMean_kwargs)
        self.centers = None

    def __intersets_distances(self, points_A, points_B, n_jobs=1, **joblib_kwargs):
        """
        Parallel computation of distances between two sets of points.
        """
        n_A, n_B = points_A.shape[0], points_B.shape[0]

        @joblib.delayed
        @joblib.wrap_non_picklable_objects
        def pickable_dist(x, y):
            return self.metric.dist(x, y)

        pool = joblib.Parallel(n_jobs=n_jobs, **joblib_kwargs)
        out = pool(
            pickable_dist(points_A[i, :], points_B[j, :])
            for i in range(n_A)
            for j in range(n_B)
        )

        # assuming numpy backend
        return np.array(out).reshape((n_A, n_B))

    def fit(
        self,
        points,
        n_centers,
        n_jobs=1,
        max_iter=100,
        init_centers="from_points",
        kernel="flat",
        **joblib_kwargs
    ):
        """
        Fit centers in all the input points to find 'n_centers' number of clusters.
        """

        @joblib.delayed
        @joblib.wrap_non_picklable_objects
        def pickable_mean(points, weights):
            return self.mean.fit(points, weights=weights).estimate_

        # 'from_points' is the default initialization for centers used
        if init_centers == "from_points":
            n_points = points.shape[0]
            centers = points[np.random.randint(n_points, size=n_centers), :]
        if init_centers == "random_uniform":
            n_centers = centers.shape[0]
            centers = self.manifold.random_uniform(n_samples=n_centers)

        for i in range(max_iter):
            dists = self.__intersets_distances(
                centers, points, n_jobs=n_jobs, **joblib_kwargs
            )

            # assuming the use of 'flat' kernel
            if kernel == "flat":
                weights = np.ones_like(dists)

            weights[dists > self.bandwidth] = 0.0
            weights = weights / weights.sum(axis=1, keepdims=1)

            points_to_average, nonzero_weights = [], []

            for j in range(n_centers):
                points_to_average += [
                    points[np.where(weights[j, :] > 0)[0], :],
                ]
                nonzero_weights += [
                    weights[j, :].nonzero()[0],
                ]

            # compute Frechet means in parallel
            pool = joblib.Parallel(n_jobs=n_jobs, **joblib_kwargs)
            out = pool(
                pickable_mean(points_to_average[j], nonzero_weights[j])
                for j in range(centers.shape[0])
            )

            new_centers = np.array(out)

            displacements = [
                self.metric.dist(centers[j], new_centers[j]) for j in range(n_centers)
            ]
            centers = new_centers

            if (np.array(displacements) < self.tol).all():
                break

        self.centers = centers

    def predict(self, points):
        """
        Predict the closest cluster each point in 'points' belongs to
        """

        if self.centers is None:
            raise Exception("Not fitted")
        else:
            out = []
            for i in range(points.shape[0]):
                j_closest_center = self.metric.closest_neighbor_index(
                    points[i, :], self.centers
                )
                out.append(self.centers[j_closest_center, :])

            return np.array(out)
