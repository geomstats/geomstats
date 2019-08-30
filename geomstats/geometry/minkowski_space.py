"""
Minkowski space.
"""


from geomstats.manifold import Manifold
from geomstats.riemannian_metric import RiemannianMetric
import geomstats.backend as gs


class MinkowskiSpace(Manifold):
    """Class for Minkowski Space."""

    def __init__(self, dimension):
        assert isinstance(dimension, int) and dimension > 0
        self.dimension = dimension
        self.metric = MinkowskiMetric(dimension)

    def belongs(self, point):
        """
        Evaluate if a point belongs to the Minkowski space.

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
        Sample in the Minkowski space with the uniform distribution.

        Returns
        -------
        points : array-like, shape=[n_samples, dimension]
                 Sampled points.
        """
        size = (n_samples, self.dimension)
        point = bound * gs.random.rand(*size) * 2 - 1

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

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]
        """
        inner_prod_mat = gs.eye(self.dimension-1, self.dimension-1)
        first_row = gs.array([0.] * (self.dimension - 1))
        first_row = gs.to_ndarray(first_row, to_ndim=2, axis=1)
        inner_prod_mat = gs.vstack([gs.transpose(first_row),
                                    inner_prod_mat])

        first_column = gs.array([-1.] + [0.] * (self.dimension - 1))
        first_column = gs.to_ndarray(first_column, to_ndim=2, axis=1)
        inner_prod_mat = gs.hstack([first_column,
                                    inner_prod_mat])

        return inner_prod_mat

    def exp(self, tangent_vec, base_point):
        """
        The Riemannian exponential is the addition in the Minkowski space.

        Parameters
        ----------
        tangent_vec: array-like, shape=[n_samples, dimension]
                                 or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        return base_point + tangent_vec

    def log(self, point, base_point):
        """
        The Riemannian logarithm is the subtraction in the Minkowski space.

        Parameters
        ----------
        point: array-like, shape=[n_samples, dimension]
                           or shape=[1, dimension]

        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]
        """
        point = gs.to_ndarray(point, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        return point - base_point

    def mean(self, points, weights=None):
        """
        The Frechet mean of (weighted) points is the weighted average of
        the points in the Minkowski space.

        Parameters
        ----------
        points: array-like, shape=[n_samples, dimension]

        weights: array-like, shape=[n_samples, 1], optional
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
