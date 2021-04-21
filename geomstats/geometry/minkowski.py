"""Minkowski space."""

import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class Minkowski(Manifold):
    """Class for Minkowski space.

    Parameters
    ----------
    dim : int
       Dimension of Minkowski space.
    """

    def __init__(self, dim):
        super(Minkowski, self).__init__(dim=dim)
        self.metric = MinkowskiMetric(dim)

    def belongs(self, point):
        """Evaluate if a point belongs to the Minkowski space.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the Minkowski space.
        """
        point_dim = point.shape[-1]
        belongs = point_dim == self.dim
        if gs.ndim(point) == 2:
            belongs = gs.tile([belongs], (point.shape[0],))

        return belongs

    def random_point(self, n_samples=1, bound=1.):
        """Sample in the Minkowski space from the uniform distribution.

        Parameters
        ----------
        n_samples: int
            Number of samples.
            Optional, default: 1
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1

        Returns
        -------
        points : array-like, shape=[..., dim]
            Sample.
        """
        size = (self.dim,)
        if n_samples != 1:
            size = (n_samples, self.dim)
        point = bound * gs.random.rand(*size) * 2 - 1

        return point


class MinkowskiMetric(RiemannianMetric):
    """Class for the pseudo-Riemannian Minkowski metric.

    The metric is flat: the inner product is independent of the base point.

    Parameters
    ----------
    dim : int
        Dimension of the Minkowski space.
    """

    def __init__(self, dim):
        super(MinkowskiMetric, self).__init__(dim=dim, signature=(1, dim - 1))

    def metric_matrix(self, base_point=None):
        """Compute the inner product matrix, independent of the base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        inner_prod_mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        p, q = self.signature
        diagonal = gs.array([-1.] * p + [1.] * q)
        return from_vector_to_diagonal_matrix(diagonal)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[..., dim]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point: array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        p, q = self.signature
        diagonal = gs.array([-1.] * p + [1.] * q)
        return gs.einsum(
            '...i,...i->...', diagonal * tangent_vec_a, tangent_vec_b)

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential of `tangent_vec` at `base_point`.

        The Riemannian exponential is the addition in the Minkowski space.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Riemannian exponential.
        """
        exp = base_point + tangent_vec
        return exp

    def log(self, point, base_point, **kwargs):
        """Compute the Riemannian logarithm of `point` at `base_point`.

        The Riemannian logarithm is the subtraction in the Minkowski space.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point.
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Riemannian logarithm.
        """
        log = point - base_point
        return log
