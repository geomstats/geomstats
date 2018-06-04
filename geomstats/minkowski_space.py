"""
Minkowski space.
"""

import geomstats.backend as gs

from geomstats.manifold import Manifold
from geomstats.riemannian_metric import RiemannianMetric


class MinkowskiSpace(Manifold):
    """Class for Minkowski Space."""

    def __init__(self, dimension):
        assert isinstance(dimension, int) and dimension > 0
        self.dimension = dimension
        self.metric = MinkowskiMetric(dimension)

    def belongs(self, point):
        """
        Evaluate if a point belongs to the Minkowski space.
        """
        point = gs.to_ndarray(point, to_ndim=2)
        n_points, point_dim = point.shape
        belongs = point_dim == self.dimension
        belongs = gs.repeat(belongs, repeats=n_points, axis=0)
        belongs = gs.to_ndarray(belongs, to_ndim=2, axis=1)

        return belongs

        point_dim = point.shape[-1]
        return point_dim == self.dimension

    def random_uniform(self, n_samples=1):
        """
        Sample in the Minkowski space with the uniform distribution.
        """
        size = (n_samples, self.dimension)
        point = gs.random.rand(*size) * 2 - 1

        return point


class MinkowskiMetric(RiemannianMetric):
    """
    Class for the pseudo-Riemannian Minkowski metric.
    The metric is flat: the inner product is independent of the base point.
    """
    def __init__(self, dimension):
        super(MinkowskiMetric, self).__init__(
                                          dimension=dimension,
                                          signature=(dimension - 1, 1, 0))

    def inner_product_matrix(self, base_point=None):
        """
        Inner product matrix, independent of the base point.
        """
        inner_prod_mat = gs.eye(self.dimension)
        inner_prod_mat[0, 0] = -1
        return inner_prod_mat

    def exp(self, tangent_vec, base_point):
        """
        The Riemannian exponential is the addition in the Minkowski space.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        return base_point + tangent_vec

    def log(self, point, base_point):
        """
        The Riemannian logarithm is the subtraction in the Minkowski space.
        """
        point = gs.to_ndarray(point, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        return point - base_point

    def mean(self, points, weights=None):
        """
        The Frechet mean of (weighted) points is the weighted average of
        the points in the Minkowski space.
        """
        mean = gs.average(points, axis=0, weights=weights)
        mean = gs.to_ndarray(mean, to_ndim=2)
        return mean
