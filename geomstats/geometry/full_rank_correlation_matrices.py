"""The manifold of full-rank correlation matrices."""

import geomstats.backend as gs
from geomstats.geometry.base import EmbeddedManifold
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricAffine


class FullRankCorrelationMatrices(EmbeddedManifold):
    """Class for the manifold of full-rank correlation matrices.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        super(FullRankCorrelationMatrices, self).__init__(
            dim=int(n * (n - 1) / 2), embedding_space=SPDMatrices(n=n),
            submersion=Matrices.diagonal, value=gs.ones(n),
            tangent_submersion=lambda v, x: Matrices.diagonal(v))
        self.n = n

    @staticmethod
    def diag_action(diagonal_vec, point):
        r"""Apply a diagonal matrix on an SPD matrices by congruence.

        The action of :math: `D` on :math: `\Sigma` is given by :math: `D
        \Sigma D. The diagonal matrix must be passed as a vector representing
        its diagonal.

        Parameters
        ----------
        diagonal_vec : array-like, shape=[..., n]
            Vector coefficient of the diagonal matrix.
        point : array-like, shape=[..., n, n]
            Symmetric Positive definite matrix.

        Returns
        -------
        mat : array-like, shape=[..., n, n]
            Symmetric matrix obtained by the action of `diagonal_vec` on
            `point`.
        """
        aux = gs.einsum('...i,...j->...ij', diagonal_vec, diagonal_vec)
        return point * aux

    @classmethod
    def from_covariance(cls, point):
        r"""Compute the correlation matrix associated to an SPD matrix.

        The correlation matrix associated to an SPD matrix (the covariance)
        :math: `\Sigma` is given by :math: `D  \Sigma D` where :math: `D` is
        the inverse square-root of the diagonal of :math: `\Sigma`.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Symmetric Positive definite matrix.

        Returns
        -------
        corr : array-like, shape=[..., n, n]
            Correlation matrix obtained by dividing all elements by the
            diagonal entries.
        """
        diag_vec = Matrices.diagonal(point) ** (-.5)
        return cls.diag_action(diag_vec, point)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample full-rank correlation matrices by projecting random SPD mats.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        bound : float
            Bound of the interval in which to sample.
            Optional, default: 1.

        Returns
        -------
        cor : array-like, shape=[n_samples, n, n]
            Sample of full-rank correlation matrices.
        """
        spd = self.embedding_space.random_point(n_samples, bound=bound)
        return self.from_covariance(spd)

    def projection(self, point):
        """Project a matrix to the space of correlation matrices.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to project.
        """
        spd = self.embedding_space.projection(point)
        return self.from_covariance(spd)

    def to_tangent(self, vector, base_point):
        """Project a matrix to the tangent space at a base point.

        The tangent space to the space of correlation matrices is the space of
        symmetric matrices with null diagonal.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Matrix to project
        base_point : array-like, shape=[..., n, n]
            Correlation matrix.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Symmetric matrix with 0 diagonal.
        """
        sym = self.embedding_space.to_tangent(vector, base_point)
        mask_diag = gs.ones_like(vector) - gs.eye(self.n)
        return sym * mask_diag


class CorrelationMatricesBundle(SPDMatrices, FiberBundle):
    """Fiber bundle to construct the quotient metric on correlation matrices.

    Correlation matrices are obtained as the quotient of the space of SPD
    matrices by the action by congruence of diagonal matrices.

    References
    ----------
    .. [TP21]  Thanwerdas, Yann, and Xavier Pennec. “Geodesics and Curvature of
               the Quotient-Affine Metrics on Full-Rank Correlation
               Matrices.” In Proceedings of Geometric Science of Information.
               Paris, France, 2021.
               https://hal.archives-ouvertes.fr/hal-03157992.
    """

    def __init__(self, n):
        super(CorrelationMatricesBundle, self).__init__(
            n=n, base=FullRankCorrelationMatrices(n),
            ambient_metric=SPDMetricAffine(n), group_dim=n,
            group_action=FullRankCorrelationMatrices.diag_action)

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
        vector = gs.einsum(
            '...ij,...ji->...i', inverse_base_point, tangent_vec)
        diagonal = gs.einsum('...ij,...j->...i', inverse_operator, vector)
        return base_point * (diagonal[..., None, :] + diagonal[..., :, None])

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        """Compute the horizontal lift wrt the affine-invariant metric.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector of the manifold of full-rank correlation matrices.
        fiber_point : array-like, shape=[..., n, n]
            SPD matrix in the fiber above point.
        base_point : array-like, shape=[..., n, n]
            Full-rank correlation matrix.

        Returns
        -------
        hor_lift : array-like, shape=[..., n, n]
            Horizontal lift of tangent_vec from point to base_point.
        """
        if fiber_point is None and base_point is not None:
            return self.horizontal_projection(tangent_vec, base_point)
        diagonal_point = Matrices.diagonal(fiber_point) ** 0.5
        lift = FullRankCorrelationMatrices.diag_action(
            diagonal_point, tangent_vec)
        hor_lift = self.horizontal_projection(lift, base_point=fiber_point)
        return hor_lift


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
