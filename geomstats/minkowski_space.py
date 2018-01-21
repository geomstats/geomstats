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


class MinkowskiMetric(RiemannianMetric):
    """
    Class for the pseudo-Riemannian Minkowski metric.
    The metric is flat: inner product independent of the reference point.
    The metric has signature (-1, n) on the (n+1)-D vector space.
    """
    def __init__(self, dimension):
        RiemannianMetric.__init__(self,
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
