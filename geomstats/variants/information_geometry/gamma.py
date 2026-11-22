"""Fisher-Rao metric on Gamma distributions given only by its metric matrix."""

import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.riemannian_metric import CoordinateRiemannianMetric


class GammaMetric(CoordinateRiemannianMetric):
    """Fisher-Rao metric on Gamma distributions given only by its metric matrix.

    The Christoffel symbols and the curvature tensors are computed from the
    metric matrix with automatic differentiation, instead of the closed forms
    of :class:`geomstats.information_geometry.gamma.GammaMetric`.
    """

    def metric_matrix(self, base_point):
        """Compute the inner-product matrix.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., 2, 2]
            Inner-product matrix.
        """
        kappa, gamma = base_point[..., 0], base_point[..., 1]
        mat_diag = gs.stack(
            [gs.polygamma(1, kappa) - 1 / kappa, kappa / gamma**2], axis=-1
        )
        return from_vector_to_diagonal_matrix(mat_diag)
