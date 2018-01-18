"""
Computations on the (n+1)-dimensional Minkowski space.
"""

import numpy as np

import Manifold
import RiemannianMetric


class MinkowskiMetric(RiemannianMetric):
    """
    Class for the pseudo-Riemannian Minkowski metric.
    The metric is flat: inner product independent of the reference point.
    The metric has signature (-1, n) on the (n+1)-D vector space.
    """

    def riemannian_inner_product(self, vector_a, vector_b):
        """Minkowski inner product."""
        return np.dot(vector_a, vector_b) - 2 * vector_a[0] * vector_b[0]

    def riemannian_squared_norm(self, vector):
        """Squared norm associated to the inner product."""
        sq_norm = self.riemannian_inner_product(vector, vector)

        return sq_norm


class MinkowskiSpace(Manifold):
    """The Minkowski Space."""

    def __init__(self, dimension):
        Manifold.__init__(dimension)
        self.riemannian_metric = MinkowskiMetric()
