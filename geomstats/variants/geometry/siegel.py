"""Variants of the metric on the Siegel space."""

import geomstats.backend as gs
from geomstats.geometry.complex_matrices import ComplexMatrixOperations
from geomstats.geometry.hermitian_matrices import powermh
from geomstats.geometry.matrices import MatrixOperations
from geomstats.geometry.siegel import SiegelMetric


class MatrixFunctionSiegelMetric(SiegelMetric):
    """Siegel metric with explicit inner product and squared distance.

    Variant of :class:`geomstats.geometry.siegel.SiegelMetric` computing the
    inner product with inverse matrix powers and the squared distance from a
    cross-ratio, through a matrix square root and logarithm.
    """

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the Siegel inner-product.

        Compute the inner-product of tangent_vec_a and tangent_vec_b
        at point base_point using the Siegel Riemannian metric.
        The expression of the inner product between the vectors ``v`` and ``w``
        at point ``O`` is :math:`<v, w>_{O}
        = 1/2 * trace((I - O O^{H})^{-1} v (I - O^{H} O)^{-1} w^{H})
        + 1/2 * trace((I - O O^{H})^{-1} w (I - O^{H} O)^{-1} v^{H})
        = Re(trace((I - O O^{H})^{-1} v (I - O^{H} O)^{-1} w^{H}))`
        where H denotes the conjugate transpose operator.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        identity = gs.eye(base_point.shape[-1], dtype=base_point.dtype)

        base_point_transconj = ComplexMatrixOperations.transconjugate(base_point)
        tangent_vec_b_transconj = ComplexMatrixOperations.transconjugate(tangent_vec_b)

        aux_1 = gs.matmul(base_point, base_point_transconj)

        aux_2 = gs.matmul(base_point_transconj, base_point)

        aux_3 = identity - aux_1

        aux_4 = identity - aux_2

        inv_aux_3 = powermh(aux_3, -1)

        inv_aux_4 = powermh(aux_4, -1)

        aux_a = gs.matmul(inv_aux_3, tangent_vec_a)
        aux_b = gs.matmul(inv_aux_4, tangent_vec_b_transconj)
        trace = MatrixOperations.trace_product(aux_a, aux_b)

        return gs.real(trace)

    def squared_dist(self, point_a, point_b):
        """Compute the Siegel squared distance.

        Compute the Riemannian squared distance between point_a and point_b.

        Parameters
        ----------
        point_a : array-like, shape=[..., n, n]
            Point.
        point_b : array-like, shape=[..., n, n]
            Point.

        Returns
        -------
        squared_dist : array-like, shape=[...]
            Riemannian squared distance.
        """
        if gs.ndim(point_a) > gs.ndim(point_b):
            point_a, point_b = point_b, point_a

        identity = gs.eye(point_a.shape[-1], dtype=point_a.dtype)

        point_a_transconj = ComplexMatrixOperations.transconjugate(point_a)
        point_b_transconj = ComplexMatrixOperations.transconjugate(point_b)

        factor_1 = point_b - point_a

        factor_3 = ComplexMatrixOperations.transconjugate(factor_1)

        aux_2 = gs.matmul(point_a_transconj, point_b)

        aux_3 = gs.matmul(point_a, point_b_transconj)

        aux_4 = identity - aux_2

        aux_5 = identity - aux_3

        factor_2 = gs.linalg.inv(aux_4)

        factor_4 = gs.linalg.inv(aux_5)

        prod = gs.einsum(
            "...ij,...jk,...kl,...lm->...im", factor_1, factor_2, factor_3, factor_4
        )

        prod_power_one_half = gs.linalg.fractional_matrix_power(prod, 0.5)

        num = identity + prod_power_one_half
        den = identity - prod_power_one_half

        inv_den = gs.linalg.inv(den)

        frac = gs.matmul(num, inv_den)

        logarithm = gs.linalg.logm(frac)

        sq_dist = MatrixOperations.trace_product(logarithm, logarithm) * 0.25
        sq_dist = gs.real(sq_dist)
        return gs.maximum(sq_dist, 0)
