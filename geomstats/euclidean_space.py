"""
The Euclidean space.
"""

from geomstats.manifold import Manifold
from geomstats.riemannian_metric import RiemannianMetric
import geomstats.backend as gs


class EuclideanSpace(Manifold):
    """The Euclidean space."""

    def __init__(self, dimension):
        assert isinstance(dimension, int) and dimension > 0
        self.dimension = dimension
        self.metric = EuclideanMetric(dimension)

    def belongs(self, point):
        """
        Check if point belongs to the Euclidean space.
        """
        point_dim = point.shape[-1]
        return point_dim == self.dimension

    def random_uniform(self, n_samples=1, n_channels=1):
        """
        Sample a vector uniformly in the Euclidean space,
        with coordinates each between -1. and 1.
        """
        point = gs.random.rand(n_samples, n_channels, self.dimension) * 2 - 1
        if n_channels == 1:
            point = gs.squeeze(point, axis=1)
        if n_samples == 1:
            point = gs.squeeze(point, axis=0)
        return point


class EuclideanMetric(RiemannianMetric):
    """
    Class for the Euclidean metric.
    The metric is flat: inner product independent of the reference point.
    The metric has signature (0, n) on the n-D vector space.
    """
    def __init__(self, dimension):
        assert isinstance(dimension, int) and dimension > 0
        super(EuclideanMetric, self).__init__(
                                        dimension=dimension,
                                        signature=(dimension, 0, 0))

    def inner_product_matrix(self, base_point=None):
        """
        Euclidean inner product matrix, which is the identity matrix.

        Note: the matrix is independent of the base_point.
        """
        return gs.eye(self.dimension)

    def exp(self, tangent_vec, base_point):
        """
        The Riemannian exponential is the addition in the Euclidean space.
        """
        exp = base_point + tangent_vec
        return exp

    def log(self, point, base_point):
        """
        The Riemannian logarithm is the subtraction in the Euclidean space.
        """
        log = point - base_point
        return log

    def mean(self, points, weights=None):
        """
        Weighted mean of the points.
        """
        return gs.average(points, axis=0, weights=weights)
