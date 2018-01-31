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
        return len(point) == self.dimension

    def random_uniform(self):
        """
        Sample a vector uniformly in the Euclidean space,
        with coordinates each between -1. and 1.
        """
        point = np.random.rand(self.dimension) * 2 - 1
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
        return base_point + tangent_vec

    def log(self, point, base_point):
        """
        The Riemannian logarithm is the subtraction in the Euclidean space.
        """
        return point - base_point

    def mean(self, points, weights=None):
        """
        Weighted mean of the points.
        """
        return np.average(points, axis=0, weights=weights)
