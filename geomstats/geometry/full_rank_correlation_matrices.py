"""The manifold of full-rank correlation matrices."""

import math

import geomstats.backend as gs
from geomstats.geometry.base import EmbeddedManifold
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricAffine


class CorrelationMatricesBundle(SPDMatrices, FiberBundle):
    def __init__(self, n):
        super(CorrelationMatricesBundle, self).__init__(
            n=n, base=FullRankCorrelationMatrices(n),
            ambient_metric=SPDMetricAffine(n))

    @staticmethod
    def riemannian_submersion(point):
        """Compute the correlation matrix associated to an SPD matrix.

                Parameters
                ----------
                point : array-like, shape=[..., n, n]
                    SPD matrix.

                Returns
                -------
                cor : array_like, shape=[..., n, n]
                    Full rank correlation matrix.
                """
        diagonal = Matrices.diagonal(point) ** (-.5)
        aux = gs.einsum('...i,...j->...ij', diagonal, diagonal)
        return point * aux

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        """Compute the differential of the submersion.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        result : array-like, shape=[..., n, n]
        """
        diagonal_bp = Matrices.diagonal(base_point)
        diagonal_tv = Matrices.diagonal(tangent_vec)

        diagonal = diagonal_tv / diagonal_bp
        aux = base_point * (diagonal[..., None, :] + diagonal[..., :, None])
        mat = tangent_vec - .5 * aux
        return FullRankCorrelationMatrices.diag_action(
            diagonal_bp ** (-.5), mat)

    def vertical_projection(self, tangent_vec, base_point, **kwargs):
        """Compute the vertical projection wrt the affine-invariant metric.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        ver : array-like, shape=[..., n, n]
            Vertical projection.
        """
        n = self.n
        inverse_base_point = GeneralLinear.inverse(base_point)
        operator = gs.eye(n) + base_point * inverse_base_point
        inverse_operator = GeneralLinear.inverse(operator)
        vector = gs.einsum('...ij,...ji->...i',
                           inverse_base_point, tangent_vec)
        inverse_operator_vector = Matrices.mul(inverse_operator, vector)
        return FullRankCorrelationMatrices.diag_action(
            inverse_operator_vector, base_point)

    def horizontal_lift(self, tangent_vec, point=None, base_point=None):
        """Compute the horizontal lift wrt the affine-invariant metric.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector of the manifold of full-rank correlation matrices.
        point : array-like, shape=[..., n, n]
            SPD matrix in the fiber above point.
        base_point : array-like, shape=[..., n, n]
            Full-rank correlation matrix.

        Returns
        -------
        hor_lift : array-like, shape=[..., n, n]
            Horizontal lift of tangent_vec from point to base_point.
        """
        if point is None and base_point is not None:
            return self.horizontal_projection(tangent_vec, base_point)
        diagonal_point = Matrices.diagonal(point) ** 0.5
        lift = FullRankCorrelationMatrices.diag_action(
            diagonal_point, tangent_vec)
        hor_lift = self.horizontal_projection(lift, base_point=point)
        return hor_lift


class FullRankCorrelationMatrices(EmbeddedManifold):
    """Class for the manifold of full-rank correlation matrices.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        super(FullRankCorrelationMatrices, self).__init__(
            dim=int(n * (n-1) / 2), embedding_space=SPDMatrices(n=n),
            submersion=Matrices.diagonal, value=gs.ones(n),
            tangent_submersion=lambda v, x: Matrices.diagonal(v))
        self.n = n

    @staticmethod
    def diag_action(diagonal_vec, point):
        aux = gs.einsum('...i,...j->...ij', diagonal_vec, diagonal_vec)
        return point * aux

    @classmethod
    def from_covariance(cls, point):
        diag_vec = Matrices.diagonal(point) ** (-.5)
        return cls.diag_action(diag_vec, point)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample full-rank correlation matrices by projecting random SPD mats.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        cor : array-like, shape=[n_samples, n, n]
            Sample of full-rank correlation matrices.
        """
        spd = self.embedding_space.random_point(n_samples)
        return self.from_covariance(spd)

    def projection(self, point):
        spd = self.embedding_space.projection(point)
        return self.from_covariance(spd)

    def to_tangent(self, vector, base_point):
        sym = self.embedding_space.to_tangent(vector, base_point)
        mask_diag = gs.ones_like(vector) - gs.eye(self.n)
        return sym * mask_diag


class FullRankCorrelationAffineQuotientMetric(QuotientMetric):
    """Class for the quotient of the affine-invariant metric.

    The affine-invariant metric on SPD matrices is invariant under the
    action of diagonal matrices, thus it induces a quotient metric on the
    manifold of full-rank correlation matrices.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        super(FullRankCorrelationAffineQuotientMetric, self).__init__(
            fiber_bundle=CorrelationMatricesBundle(n=n))

    # def inner_product(
    #         self, tangent_vec_a, tangent_vec_b, base_point=None, point=None):
    #     """Compute the inner product of the affine-quotient metric.
    #
    #     Parameters
    #     ----------
    #     tangent_vec_a : array-like, shape=[..., n, n]
    #         Tangent vector to the manifold of full-rank correlation matrices.
    #     tangent_vec_b : array-like, shape=[..., n, n]
    #         Tangent vector to the manifold of full-rank correlation matrices.
    #     base_point : array-like, shape=[..., n, n]
    #         Full-rank correlation matrix.
    #
    #     Returns
    #     -------
    #     inner_product : array-like, shape=[...]
    #         Inner product of tangent_vec_a and tangent_vec_b at base_point.
    #     """
    #     affine_part = self.ambient_metric.inner_product(
    #         tangent_vec_a, tangent_vec_b, base_point)
    #     n = gs.shape(base_point)[-1]
    #     inverse_base_point = GeneralLinear.inverse(base_point)
    #     diagonal_a = gs.einsum('...ij,...ji->...i',
    #                            inverse_base_point, tangent_vec_a)
    #     diagonal_b = gs.einsum('...ij,...ji->...i',
    #                            inverse_base_point, tangent_vec_b)
    #     operator = gs.eye(n) + base_point * inverse_base_point
    #     inverse_operator = GeneralLinear.inverse(operator)
    #     inverse_operator_diagonal_b = gs.einsum('...ij,...j->...ij',
    #                                             inverse_operator, diagonal_b)
    #     other_part = 2 * gs.einsum('...i,...ij->...',
    #                                diagonal_a, inverse_operator_diagonal_b)
    #     inner_product = affine_part + other_part
    #     return inner_product
