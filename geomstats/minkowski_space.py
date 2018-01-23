"""
Computations on the (n+1)-dimensional Minkowski space.
"""

import numpy as np

from geomstats.manifold import Manifold
from geomstats.riemannian_metric import RiemannianMetric


class MinkowskiSpace(Manifold):
    """The Minkowski Space."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.metric = MinkowskiMetric(dimension)

    def belongs(self, point):
        """
        Check if point belongs to the Minkowski space.
        """
        return len(point) == self.dimension

    def random_uniform(self):
        """
        Sample a vector uniformly in the Minkowski space,
        with coordinates each between -1. and 1.
        """
        point = np.random.rand(self.dimension) * 2 - 1
        return point


class MinkowskiMetric(RiemannianMetric):
    """
    Class for the pseudo-Riemannian Minkowski metric.
    The metric is flat: inner product independent of the reference point.
    The metric has signature (-1, n) on the (n+1)-D vector space.
    """
    def __init__(self, dimension):
        super(MinkowskiMetric, self).__init__(
                                          dimension=dimension,
                                          signature=(dimension - 1, 1, 0))

    def inner_product_matrix(self, base_point=None):
        """
        Minkowski inner product matrix.

        Note: the matrix is independent on the base_point.
        """
        inner_prod_mat = np.eye(self.dimension)
        inner_prod_mat[0, 0] = -1
        return inner_prod_mat

    def exp(self, tangent_vec, base_point):
        """
        The Riemannian exponential is the addition in the Minkowski space.
        """
        return base_point + tangent_vec

    def log(self, point, base_point):
        """
        The Riemannian logarithm is the subtraction in the Minkowski space.
        """
        return point - base_point

    def mean(self, points, weights=None):
        """
        Weighted mean of the points.
        """
        return np.average(points, axis=0, weights=weights)
