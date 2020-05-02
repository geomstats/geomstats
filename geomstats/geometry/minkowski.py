"""Minkowski space."""

import geomstats.backend as gs
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

    def random_uniform(self, n_samples=1, bound=1.):
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
        super(MinkowskiMetric, self).__init__(
            dim=dim,
            signature=(dim - 1, 1, 0))

    def inner_product_matrix(self, base_point=None):
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
        inner_prod_mat = gs.eye(self.dim - 1, self.dim - 1)
        first_row = gs.array([0.] * (self.dim - 1))
        first_row = gs.to_ndarray(first_row, to_ndim=2, axis=1)
        inner_prod_mat = gs.vstack(
            [gs.transpose(first_row), inner_prod_mat])

        first_column = gs.array([-1.] + [0.] * (self.dim - 1))
        first_column = gs.to_ndarray(first_column, to_ndim=2, axis=1)
        inner_prod_mat = gs.hstack([first_column, inner_prod_mat])

        return inner_prod_mat

    def exp(self, tangent_vec, base_point):
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

    def log(self, point, base_point):
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
