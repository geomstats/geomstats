"""Fisher-Rao metric on Dirichlet distributions given only by its metric matrix."""

import geomstats.backend as gs
from geomstats._einsum import scalarmatmul
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.riemannian_metric import CoordinateRiemannianMetric


class DirichletMetric(CoordinateRiemannianMetric):
    """Fisher-Rao metric on Dirichlet distributions given only by its metric matrix.

    The Christoffel symbols and the curvature tensors are computed from the
    metric matrix with automatic differentiation, instead of the closed forms
    of :class:`geomstats.information_geometry.dirichlet.DirichletMetric`.
    """

    def metric_matrix(self, base_point):
        """Compute the inner-product matrix.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        batch_shape = base_point.shape[:-1]

        mat_ones = gs.ones(batch_shape + (self._space.dim, self._space.dim))
        poly_sum = gs.polygamma(1, gs.sum(base_point, axis=-1))
        mat_diag = from_vector_to_diagonal_matrix(gs.polygamma(1, base_point))

        return mat_diag - scalarmatmul(poly_sum, mat_ones)
