"""
Euclidean space.
"""

import geomstats.backend as gs

from geomstats.manifold import Manifold
from geomstats.riemannian_metric import RiemannianMetric


class EuclideanSpace(Manifold):
    """
    Class for Euclidean spaces.

    By definition, a Euclidean space is a vector space of a given
    dimension, equipped with a Euclidean metric.
    """

    def __init__(self, dimension):
        assert isinstance(dimension, int) and dimension > 0
        self.dimension = dimension
        self.metric = EuclideanMetric(dimension)

    def belongs(self, point):
        """
        Evaluate if a point belongs to the Euclidean space.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
                Input points.

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
        """
        point = gs.to_ndarray(point, to_ndim=2)
        n_points, point_dim = point.shape
        belongs = point_dim == self.dimension
        belongs = gs.to_ndarray(belongs, to_ndim=1)
        belongs = gs.to_ndarray(belongs, to_ndim=2, axis=1)
        belongs = gs.tile(belongs, (n_points, 1))

        return belongs

    def random_uniform(self, n_samples=1, bound=1.):
        """
        Sample in the Euclidean space with the uniform distribution.
        """
        size = (n_samples, self.dimension)
        point = bound * (gs.random.rand(*size) - 0.5) * 2

        return point


class EuclideanMetric(RiemannianMetric):
    """
    Class for Euclidean metrics.

    As a Riemannian metric, the Euclidean metric is:
    - flat: the inner product is independent of the base point.
    - positive definite: it has signature (0, dimension),
    where dimension is the dimension of the Euclidean space.
    """
    def __init__(self, dimension):
        assert isinstance(dimension, int) and dimension > 0
        super(EuclideanMetric, self).__init__(
                                        dimension=dimension,
                                        signature=(dimension, 0, 0))

    def inner_product_matrix(self, base_point=None):
        """
        Inner product matrix, independent of the base point.
        """
        mat = gs.eye(self.dimension)
        mat = gs.to_ndarray(mat, to_ndim=3)
        return mat

    def exp(self, tangent_vec, base_point):
        """
        The Riemannian exponential is the addition in the Euclidean space.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        exp = base_point + tangent_vec
        return exp

    def log(self, point, base_point):
        """
        The Riemannian logarithm is the subtraction in the Euclidean space.
        """
        point = gs.to_ndarray(point, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        log = point - base_point
        return log

    def mean(self, points, weights=None):
        """
        The Frechet mean of (weighted) points computed with the
        Euclidean metric is the weighted average of the points
        in the Euclidean space.
        """
        if isinstance(points, list):
            points = gs.vstack(points)
        points = gs.to_ndarray(points, to_ndim=2)
        n_points = gs.shape(points)[0]

        if isinstance(weights, list):
            weights = gs.vstack(weights)
        elif weights is None:
            weights = gs.ones((n_points,))

        weighted_points = gs.einsum('n,nj->nj', weights, points)
        mean = (gs.sum(weighted_points, axis=0)
                / gs.sum(weights))
        mean = gs.to_ndarray(mean, to_ndim=2)
        return mean
