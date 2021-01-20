"""The manifold of full-rank correlation matrices."""

import math

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricAffine
from geomstats.geometry.symmetric_matrices import SymmetricMatrices

EPSILON = 1e-6
TOLERANCE = 1e-12

class FullRankCorrelationMatrices(SPDMatrices, EmbeddedManifold):
    """Class for the manifold of full-rank correlation matrices.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        super(FullRankCorrelationMatrices, self).__init__(
            n=n,
            dim=int(n * (n-1) / 2),
            embedding_manifold=GeneralLinear(n=n))

    def belongs(self, mat, atol=TOLERANCE):
        """Check if a matrix is a full rank correlation matrix.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix to be checked.
        atol : float
            Tolerance.
            Optional, default: TOLERANCE.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if mat is an SPD matrix."""
        is_spd = super(FullRankCorrelationMatrices, self).belongs(mat, atol)
        diagonal = gs.einsum('...ii->...i', mat)
        is_diag_one = gs.all(diagonal == 1)
        belongs = is_spd and is_diag_one
        return belongs

    def random_uniform(self, n_samples=1):
        """
        Sample of full-rank correlation matrices from a 'uniform' distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        cor : array-like, shape=[n_samples, n, n]
            Sample of full-rank correlation matrices.
        """
        spd = super(FullRankCorrelationMatrices, self).random_uniform(
            n_samples=n_samples)
        cor = FullRankCorrelationMatrices.from_spd_to_correlation(spd)
        return cor


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
            fiber_bundle=SPDMatrices(n=n),
            ambient_metric=SPDMetricAffine(n=n))

    def inner_product(
            self, tangent_vec_a, tangent_vec_b, base_point, point=None):
        """Compute the inner product of the affine-quotient metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector to the manifold of full-rank correlation matrices.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector to the manifold of full-rank correlation matrices.
        base_point : array-like, shape=[..., n, n]
            Full-rank correlation matrix.

        Returns
        -------
        inner_product : array-like, shape=[...]
            Inner product of tangent_vec_a and tangent_vec_b at base_point.
        """
        affine_part = self.ambient_metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point)
        n = gs.shape(base_point)[-1]
        inverse_base_point = GeneralLinear.inverse(base_point)
        diagonal_a = gs.einsum('...ij,...ji->...i',
                               inverse_base_point, tangent_vec_a)
        diagonal_b = gs.einsum('...ij,...ji->...i',
                               inverse_base_point, tangent_vec_b)
        operator = gs.eye(n) + base_point * inverse_base_point
        inverse_operator = GeneralLinear.inverse(operator)
        inverse_operator_diagonal_b = gs.einsum('...ij,...j->...ij',
                                                inverse_operator, diagonal_b)
        other_part = 2 * gs.einsum('...i,...ij->...',
                                   diagonal_a, inverse_operator_diagonal_b)
        inner_product = affine_part + other_part
        return inner_product