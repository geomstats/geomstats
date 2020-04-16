"""Euclidean space."""

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class Euclidean(Manifold):
    """Class for Euclidean spaces.

    By definition, a Euclidean space is a vector space of a given
    dimension, equipped with a Euclidean metric.
    """

    def __init__(self, dim):
        super(Euclidean, self).__init__(dim=dim)
        self.metric = EuclideanMetric(dim)

    def belongs(self, point):
        """Evaluate if a point belongs to the Euclidean space.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dim]
                Input points.

        Returns
        -------
        belongs : array-like, shape=[n_samples,]
        """
        point_dim = point.shape[-1]
        belongs = point_dim == self.dim
        if gs.ndim(point) == 2:
            belongs = gs.tile([belongs], (point.shape[0],))

        return belongs

    def random_uniform(self, n_samples=1, bound=1.):
        """Sample in the Euclidean space with the uniform distribution.

        Parameters
        ----------
        n_samples: int, optional
        bound: float, optional

        Returns
        -------
        point : array-like, shape=[n_samples, dim]
        """
        size = (self.dim,)
        if n_samples != 1:
            size = (n_samples, self.dim)
        point = bound * (gs.random.rand(*size) - 0.5) * 2

        return point


class EuclideanMetric(RiemannianMetric):
    """Class for Euclidean metrics.

    As a Riemannian metric, the Euclidean metric is:
    - flat: the inner product is independent of the base point.
    - positive definite: it has signature (dimension, 0, 0),
    where dimension is the dimension of the Euclidean space.
    """

    def __init__(self, dim):
        super(EuclideanMetric, self).__init__(
            dim=dim, signature=(dim, 0, 0))

    def inner_product_matrix(self, base_point=None):
        """Compute inner product matrix, independent of the base point.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dim]

        Returns
        -------
        inner_prod_mat: array-like, shape=[n_samples, dim, dim]
        """
        mat = gs.eye(self.dim)
        return mat

    def exp(self, tangent_vec, base_point):
        """Compute exp map of a base point in tangent vector direction.

        The Riemannian exponential is vector addition in the Euclidean space.

        Parameters
        ----------
        tangent_vec: array-like, shape=[n_samples, dim]
                                 or shape=[1, dim]

        base_point: array-like, shape=[n_samples, dim]
                                or shape=[1, dim]

        Returns
        -------
        exp: array-like, shape=[n_samples, dim]
                          or shape-[n_samples, dim]
        """
        exp = base_point + tangent_vec
        return exp

    def log(self, point, base_point):
        """Compute log map using a base point and other point.

        The Riemannian logarithm is the subtraction in the Euclidean space.

        Parameters
        ----------
        point: array-like, shape=[n_samples, dim]
                           or shape=[1, dim]

        base_point: array-like, shape=[n_samples, dim]
                                or shape=[1, dim]

        Returns
        -------
        log: array-like, shape=[n_samples, dim]
                          or shape-[n_samples, dim]
        """
        log = point - base_point
        return log
