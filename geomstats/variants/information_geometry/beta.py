"""Fisher-Rao metric on Beta distributions given only by its metric matrix."""

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import CoordinateRiemannianMetric
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


class BetaMetric(CoordinateRiemannianMetric):
    """Fisher-Rao metric on Beta distributions given only by its metric matrix.

    The Christoffel symbols and the curvature tensors are computed from the
    metric matrix with automatic differentiation, instead of the closed forms
    of :class:`geomstats.information_geometry.beta.BetaMetric`.
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
        param_a = base_point[..., 0]
        param_b = base_point[..., 1]
        trigamma_sum = gs.polygamma(1, param_a + param_b)
        vector = gs.stack(
            [
                gs.polygamma(1, param_a) - trigamma_sum,
                -trigamma_sum,
                gs.polygamma(1, param_b) - trigamma_sum,
            ],
            axis=-1,
        )
        return SymmetricMatrices.matrix_representation(vector)
