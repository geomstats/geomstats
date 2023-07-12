"""The manifold of full-rank correlation matrices.

Lead author: Yann Thanwerdas.
"""

import geomstats.backend as gs
from geomstats.geometry.base import LevelSet
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.lower_triangular_matrices import StrictlyLowerTriangularMatrices
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDAffineMetric, SPDMatrices


class FullRankCorrelationMatrices(LevelSet):
    """Class for the manifold of full-rank correlation matrices.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n, equip=True):
        self.n = n
        super().__init__(dim=int(n * (n - 1) / 2), equip=equip)

    def _define_embedding_space(self):
        return SPDMatrices(n=self.n)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return FullRankCorrelationAffineQuotientMetric

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]

        Returns
        -------
        submersed_point : array-like, shape=[..., n]
        """
        return Matrices.diagonal(point) - gs.ones(self.n)

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
        point : Ignored.

        Returns
        -------
        submersed_vector : array-like, shape=[..., n]
        """
        submersed_vector = Matrices.diagonal(vector)
        if point is not None and point.ndim > vector.ndim:
            return gs.broadcast_to(submersed_vector, point.shape[:-1])

        return submersed_vector

    @staticmethod
    def diag_action(diagonal_vec, point):
        r"""Apply a diagonal matrix on an SPD matrices by congruence.

        The action of :math:`D` on :math:`\Sigma` is given by :math:`D
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
        return point * gs.outer(diagonal_vec, diagonal_vec)

    @classmethod
    def from_covariance(cls, point):
        r"""Compute the correlation matrix associated to an SPD matrix.

        The correlation matrix associated to an SPD matrix (the covariance)
        :math:`\Sigma` is given by :math:`D  \Sigma D` where :math:`D` is
        the inverse square-root of the diagonal of :math:`\Sigma`.

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
        diag_vec = Matrices.diagonal(point) ** (-0.5)
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


class CorrelationMatricesBundle(FiberBundle):
    """Fiber bundle to construct the quotient metric on correlation matrices.

    Correlation matrices are obtained as the quotient of the space of SPD
    matrices by the action by congruence of diagonal matrices.

    References
    ----------
    .. [TP21] Thanwerdas, Yann, and Xavier Pennec. “Geodesics and Curvature of
        the Quotient-Affine Metrics on Full-Rank CorrelationMatrices.”
        In Proceedings of Geometric Science of Information.
        Paris, France, 2021.
        https://hal.archives-ouvertes.fr/hal-03157992.
    """

    def __init__(self, total_space):
        super().__init__(
            total_space=total_space,
            group_dim=total_space.n,
            group_action=FullRankCorrelationMatrices.diag_action,
        )

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
        diagonal = Matrices.diagonal(point) ** (-0.5)
        return point * gs.outer(diagonal, diagonal)

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
        mat = tangent_vec - 0.5 * aux
        return self.group_action(diagonal_bp ** (-0.5), mat)

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
        n = self.total_space.n
        inverse_base_point = GeneralLinear.inverse(base_point)
        operator = gs.eye(n) + base_point * inverse_base_point
        inverse_operator = GeneralLinear.inverse(operator)
        vector = gs.einsum("...ij,...ji->...i", inverse_base_point, tangent_vec)
        diagonal = gs.einsum("...ij,...j->...i", inverse_operator, vector)
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
        lift = self.group_action(diagonal_point, tangent_vec)
        return self.horizontal_projection(lift, base_point=fiber_point)


class FullRankCorrelationAffineQuotientMetric(QuotientMetric):
    """Class for the quotient of the affine-invariant metric.

    The affine-invariant metric on SPD matrices is invariant under the
    action of diagonal matrices, thus it induces a quotient metric on the
    manifold of full-rank correlation matrices.
    """

    def __init__(self, space, total_space=None):
        if total_space is None:
            total_space = SPDMatrices(space.n, equip=False)
            total_space.equip_with_metric(SPDAffineMetric)

        super().__init__(
            space=space,
            fiber_bundle=CorrelationMatricesBundle(total_space),
        )


class FullRankCorrelationEuclideanCholeskyMetric(PullbackDiffeoMetric):
    """Class for the Euclidean-Cholesky metric on correlation matrices.

    The Cholesky decomposition of a full-rank correlation matrix is a lower
    triangular matrices with positive coefficients on the diagonal.
    Multiplying on the left by the inverse of the positive diagonal gives a
    lower triangular matrix with unit diagonal. Forgetting the diagonal, this
    composition is a smooth diffeomorphism from full-rank correlation matrices
    to strictly lower triangular matrices, which is a Euclidean space. Thus,
    any inner product on strictly lower triangular matrices defines a flat
    complete Riemannian metric on full-rank correlation matrices by pullback.

    Parameters
    ----------
    n : int
        Dimension of correlation matrices.

    References
    ----------
    .. [TP2022] Thanwerdas, Pennec. "Theoretically and computationally
    convenient geometries on full-rank correlation matrices"
    https://arxiv.org/abs/2201.06282
    """

    def __init__(self, n):
        super().__init__(
            dim=int(n * (n - 1) / 2),
        )
        self._embedding_metric = StrictlyLowerTriangularMatrices(n).metric
        self.n = n

    def diffeomorphism(self, base_point):
        r"""Compute the diffeomorphism :math:`\Theta` at base_point.

        For a full-rank correlation matrix :math:`C`, :math:`\Theta(C)` is
        the strictly lower triangular matrix defined by
        :math:`\Theta(C)=Diag(Chol(C))^{-1}Chol(C)-I_n`.

        References
        ----------
        [TP2022] Section 4.2

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Base point, full-rank correlation matrix.

        Returns
        -------
        image : array-like, shape=[..., n, n]
            Image, strictly lower triangular matrix.
        """
        chol = gs.linalg.cholesky(base_point)
        diag = Matrices.to_diagonal(chol)
        diag_inv = GeneralLinear.inverse(diag)
        image = gs.matmul(diag_inv, chol) - gs.eye(self.n)
        return image

    def tangent_diffeomorphism(self, tangent_vec, base_point):
        r"""Compute the differential of the diffeomorphism.

        For a full-rank correlation matrix :math:`C` and a tangent
        vector :math:`X` (symmetric with null diagonal), this returns
        :math:`d_C\Theta(X)=\Theta(C)low(L^{-1}XL^{-\top})
        -\frac{1}{2}Diag(L^{-1}XL^{-\top})\Theta(C)+low_0(L^{-1}XL^{-\top})`
        where :math:`L=Chol(C)`, :math:`low_0` is the strictly lower
        triangular part and :math:`low=low_0+\frac{1}{2}Diag`.
        """
        chol = gs.linalg.cholesky(base_point)
        diag = Matrices.to_diagonal(chol)
        diag_inv = GeneralLinear.inverse(diag)
        diag_inv_squared = diag_inv ** 2
        chol_inv = GeneralLinear.inverse(chol)
        chol_inv_trans = Matrices.transpose(chol_inv)
        prod = gs.matmul(chol_inv, tangent_vec)
        prod = gs.matmul(prod, chol_inv_trans)
        low_prod = Matrices.to_strictly_lower_triangular(prod)
        low_sym_prod = low_prod + Matrices.to_diagonal(prod) / 2
        tangent_chol = gs.matmul(chol, low_sym_prod)
        diag_tangent_chol = Matrices.to_diagonal(tangent_chol)
        term_a = gs.matmul(diag_inv, tangent_chol)
        term_b = diag_inv_squared * diag_tangent_chol
        term_b = gs.matmul(term_b, chol)
        return term_a - term_b + low_prod

    def inverse_diffeomorphism(self, image_point):
        r"""Compute the inverse diffeomorphism :math:`\Theta^{-1}` at image_point.

        For a strictly lower triangular matrix :math:`L`, we have
        :math:`\Theta^{-1}(L)=cor((I_n+L)(I_n+L)^\top)` where
        :math:`cor(\Sigma)=Diag(\Sigma)^{-1/2}\Sigma Diag(\Sigma)^{-1/2}`.
        """
        reduced_chol = gs.eye(self.n) + image_point
        reduced_chol_trans = Matrices.transpose(reduced_chol)
        product = gs.matmul(reduced_chol, reduced_chol_trans)
        preimage = CorrelationMatricesBundle.riemannian_submersion(product)
        return preimage

    def inverse_tangent_diffeomorphism(self, image_tangent_vec, image_point):
        r"""Compute the differential of the inverse diffeomorphism :math:`\Theta^{-1}`.

        For strictly lower triangular matrices :math:`\Gamma` and :math:`V`,
        :math:`d_\Gamma\Theta^{-1}(V)=
        d_{(I_n+\Gamma)(I_n+\Gamma)^\top}cor(2V+V\Gamma^\top+\Gamma V^\top)`.

        For a full-rank correlation matrix :math:`C` and an image tangent vector
        :math:`V` (strictly lower triangular matrix), we have
        :math:`(d_C\Theta)^{-1}(V)=(LV^\top-C Diag(LV^\top))Diag(L)
        +Diag(L)(VL^\top-Diag(LV^\top)C)` where :math:`L=Chol(C)`.
        """
        low_tri = image_point + gs.eye(self.n)
        preimage = gs.matmul(low_tri, Matrices.transpose(low_tri))
        tangent_vec_trans = Matrices.transpose(image_tangent_vec)
        pre_tangent_vec = gs.matmul(image_point, tangent_vec_trans)
        pre_tangent_vec += Matrices.transpose(pre_tangent_vec)
        pre_tangent_vec += 2 * image_tangent_vec
        pre_tangent_vec = CorrelationMatricesBundle(self.n).\
            tangent_riemannian_submersion(pre_tangent_vec, preimage)
        return pre_tangent_vec
