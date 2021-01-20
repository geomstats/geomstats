"""The manifold of full-rank correlation matrices."""

import math

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDMatrices
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

    def from_spd_to_correlation(self, spd):
        """Compute the correlation matrix associated to an SPD matrix.

        Parameters
        ----------
        spd : array-like, shape=[..., n, n]
            SPD matrix.

        Returns
        -------
        cor : array_like, shape=[..., n, n]
            Full rank correlation matrix.
        """
        diagonal = gs.einsum('...ii->...i', spd)
        cor = gs.einsum('...i,...ij->...ij', diagonal, spd)
        cor = gs.einsum('...ij,...j->...ij', cor, diagonal)
        return cor

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

