"""
The Euclidean space.
"""

import numpy as np

from geomstats.manifold import Manifold
from geomstats.riemannian_metric import RiemannianMetric


class EuclideanSpace(Manifold):
    """The Euclidean space."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.metric = EuclideanMetric(dimension)

    def belongs(self, point):
        """
        Check if point belongs to the Euclidean space.
        """
        if point.ndim == 1:
            point = np.expand_dims(point, axis=0)
        return point.shape[1] == self.dimension

    def random_uniform(self, n_samples=1):
        """
        Sample a vector uniformly in the Euclidean space,
        with coordinates each between -1. and 1.
        """
        point = np.random.rand(n_samples, self.dimension) * 2 - 1
        return point


class EuclideanMetric(RiemannianMetric):
    """
    Class for the Euclidean metric.
    The metric is flat: inner product independent of the reference point.
    The metric has signature (0, n) on the n-D vector space.
    """
    def __init__(self, dimension):
        super(EuclideanMetric, self).__init__(
                                        dimension=dimension,
                                        signature=(dimension, 0, 0))

    def inner_product_matrix(self, base_point=None):
        """
        Euclidean inner product matrix, which is the identity matrix.

        Note: the matrix is independent of the base_point.
        """
        return np.eye(self.dimension)

    def exp(self, tangent_vec, base_point):
        """
        The Riemannian exponential is the addition in the Euclidean space.
        """
        if tangent_vec.ndim == 1:
            tangent_vec = np.expand_dims(tangent_vec, axis=0)
        assert tangent_vec.ndim == 2

        if base_point.ndim == 1:
            base_point = np.expand_dims(base_point, axis=0)
        assert base_point.ndim == 2

        n_tangent_vecs, _ = tangent_vec.shape
        n_base_points, _ = base_point.shape

        assert (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1)

        n_exps = np.maximum(n_tangent_vecs, n_base_points)
        exp = np.zeros((n_exps, self.dimension))
        for i in range(n_exps):
            base_point_i = (base_point[0] if n_base_points == 1
                            else base_point[i])
            tangent_vec_i = (tangent_vec[0] if n_tangent_vecs == 1
                             else tangent_vec[i])
            exp[i] = base_point_i + tangent_vec_i

        return exp

    def log(self, point, base_point):
        """
        The Riemannian logarithm is the subtraction in the Euclidean space.
        """
        if point.ndim == 1:
            point = np.expand_dims(point, axis=0)
        assert point.ndim == 2

        if base_point.ndim == 1:
            base_point = np.expand_dims(base_point, axis=0)
        assert base_point.ndim == 2

        n_points, _ = point.shape
        n_base_points, _ = base_point.shape

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

        n_logs = np.maximum(n_points, n_base_points)
        log = np.zeros((n_logs, self.dimension))
        for i in range(n_logs):
            base_point_i = (base_point[0] if n_base_points == 1
                            else base_point[i])
            point_i = (point[0] if n_points == 1
                       else point[i])
            log[i] = point_i - base_point_i

        return log

    def mean(self, points, weights=None):
        """
        Weighted mean of the points.
        """
        return np.average(points, axis=0, weights=weights)
