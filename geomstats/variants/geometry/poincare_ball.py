"""Poincare ball metric given only by its metric matrix."""

import geomstats.backend as gs
from geomstats._einsum import scalarmatmul
from geomstats.geometry.riemannian_metric import CoordinateRiemannianMetric


class PoincareBallMetric(CoordinateRiemannianMetric):
    """Poincare ball metric given only by its metric matrix.

    The Christoffel symbols and the curvature tensors are computed from the
    metric matrix with automatic differentiation, instead of the closed forms
    of :class:`geomstats.geometry.poincare_ball.PoincareBallMetric`.
    """

    def metric_matrix(self, base_point):
        r"""Compute the inner product matrix.

        .. math::

            G(x) = \lambda_x^2 I, \quad \lambda_x = \frac{2}{1 - ||x||^2}

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        inner_prod_mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        lambda_base = (2 / (1 - gs.sum(base_point**2, axis=-1))) ** 2
        identity = gs.eye(self._space.dim, dtype=base_point.dtype)
        return scalarmatmul(lambda_base, identity)
