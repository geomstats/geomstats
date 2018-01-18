"""
Computations on the n-dimensional Euclidean space.
"""

import numpy as np
import math

import Manifold
import RiemannianMetric


class EuclideanMetric(RiemannianMetric):
    """
    Class for the Euclidean metric.
    The metric is flat: inner product independent of the reference point.
    The metric has signature (0, n) on the n-D vector space.
    """

    def riemannian_inner_product(vector_a, vector_b):
        """Euclidean inner product."""
        return np.dot(vector_a, vector_b)

    def riemannian_squared_norm(self, vector):
        """Squared norm associated to the inner product."""
        sq_norm = self.riemannian_inner_product(vector, vector)

        return sq_norm

    def riemannian_norm(self, vector):
        """
        Norm associated to the inner product,
        as the squared norm is always positive.
        """
        sq_norm = self.riemannian_squared_norm(vector, vector)
        norm = math.sqrt(sq_norm)
        return norm


class EuclideanSpace(Manifold):
    """The Euclidean space."""

    def __init__(self, dimension):
        Manifold.__init__(dimension)
        self.riemannian_metric = EuclideanMetric()
