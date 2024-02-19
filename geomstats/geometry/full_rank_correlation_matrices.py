"""The manifold of full-rank correlation matrices.

Lead authors: Yann Thanwerdas and Olivier Bisson.


References
----------
.. [T2022] Yann Thanwerdas. Riemannian and stratified
geometries on covariance and correlation matrices. Differential
Geometry [math.DG]. Université Côte d'Azur, 2022.
"""

import logging

import geomstats.backend as gs
from geomstats.geometry.base import LevelSet
from geomstats.geometry.diffeo import ComposedDiffeo, Diffeo
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.hermitian_matrices import expmh
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.open_hemisphere import (
    OpenHemispheresProduct,
    OpenHemisphereToHyperboloidDiffeo,
)
from geomstats.geometry.positive_lower_triangular_matrices import (
    UnitNormedRowsPLTDiffeo,
)
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import (
    CholeskyMap,
    SPDAffineMetric,
    SPDMatrices,
    SymMatrixLog,
    logmh,
)
from geomstats.geometry.symmetric_matrices import (
    HollowMatricesPermutationInvariantMetric,
    SymmetricHollowMatrices,
)


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
        \Sigma D`. The diagonal matrix must be passed as a vector representing
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
        :math:`\Sigma` is given by :math:`D \Sigma D` where :math:`D` is
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

    def vertical_projection(self, tangent_vec, base_point):
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
        n = self._total_space.n
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
        if base_point is not None:
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

        if not hasattr(total_space, "fiber_bundle"):
            total_space.fiber_bundle = CorrelationMatricesBundle(total_space)

        super().__init__(
            space=space,
            total_space=total_space,
        )


class PolyHyperbolicCholeskyMetric(PullbackDiffeoMetric):
    """Pullback metric via a diffeomorphism.

    Diffeormorphism between full-rank correlation matrices and
    the space of lower triangular matrices with positive diagonal
    and unit normed rows.

    Since this image space is also diffeomorphic to another space, the
    product space of successively increasing factor-dimension open hemispheres,
    we take advantage of `ComposedDiffeo` to avoid explicitly representing
    the intermediate space.

    For more details, check section 7.4.1 [T2022]_.
    """

    def __init__(self, space):
        n = space.n
        diffeos = [CholeskyMap(), UnitNormedRowsPLTDiffeo(n)]

        if n == 2:
            diffeos.append(OpenHemisphereToHyperboloidDiffeo())
            image_space = Hyperboloid(dim=1)
        else:
            image_space = OpenHemispheresProduct(n=n)

        diffeo = ComposedDiffeo(diffeos)

        super().__init__(space=space, diffeo=diffeo, image_space=image_space)


def off_map(matrix):
    """Subtract diagonal to a matrix."""
    return matrix - Matrices.to_diagonal(matrix)


class OffLogDiffeo(Diffeo):
    r"""Off-log diffeomorphism from Cor+ to Hol.

    A diffeomorphism between full-rank correlation matrices Cor+ and
    symmetric hollow matrices Hol:

    .. math::
        \operatorname{Log} = \operatorname{Off} \circ \log :
        \operatorname{Cor}^{+}(n) \longrightarrow \operatorname{Hol}(n)

    Check out chapter 8 of [T2022]_ for more details.
    """

    def __init__(self, space, atol=gs.atol, max_iter=100):
        self.space = space
        self.atol = atol
        self.max_iter = max_iter

    def _unique_diag_mat_single(self, sym_mat):
        """Find unique diagonal matrix corresponding to a Cor+ mat. That is, for all
        symmetric matrix S, there exists a unique diagonal matrix D such that
        exp(D+S) is a full-rank correlation matrix. 

        Converges in logarithmic time to the solution of the equation, no closed form.

        Parameters
        ----------
        sym_mat : array-like, shape=[n, n]

        Returns
        -------
        diag_mat : array-like, shape=[n, n]
        """
        diag_mat = gs.zeros_like(sym_mat)

        approx_cor_mat = expmh(diag_mat + sym_mat)
        if self.space.belongs(approx_cor_mat, atol=self.atol):
            return diag_mat

        for _ in range(self.max_iter):
            diag_mat = diag_mat - logmh(Matrices.to_diagonal(approx_cor_mat))
            approx_cor_mat = expmh(diag_mat + sym_mat)
            if self.space.belongs(approx_cor_mat, atol=self.atol):
                return diag_mat
        else:
            logging.warning(
                "Maximum number of iterations %d reached. The mean may be inaccurate",
                self.max_iter,
            )

        return diag_mat

    def _unique_diag_mat(self, sym_mat):
        """Find unique diagonal matrix corresponding to a Cor+ mat. That is, for all
        symmetric matrix S, there exists a unique diagonal matrix D such that
        exp(D+S) is a full-rank correlation matrix. 

        Parameters
        ----------
        sym_mat : array-like, shape=[..., n, n]

        Returns
        -------
        diag_mat : array-like, shape=[..., n, n]
        """
        if sym_mat.ndim == 2:
            return self._unique_diag_mat_single(sym_mat)

        batch_shape = sym_mat.shape[:-2]
        if len(batch_shape) == 1:
            return gs.stack(
                [self._unique_diag_mat_single(sym_mat_) for sym_mat_ in sym_mat]
            )

        mat_shape = sym_mat.shape[-2:]
        flat_sym_mat = gs.reshape(sym_mat, (-1,) + mat_shape)
        out = gs.stack(
            [self._unique_diag_mat_single(sym_mat_) for sym_mat_ in flat_sym_mat]
        )
        return gs.reshape(out, batch_shape + mat_shape)

    def diffeomorphism(self, base_point):
        """Diffeomorphism at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., n, n]
            Image point.
        """
        return off_map(matrix=logmh(mat=base_point))

    def inverse_diffeomorphism(self, image_point):
        r"""Inverse diffeomorphism at image point.

        :math:`f^{-1}: N \rightarrow M`

        Parameters
        ----------
        image_point : array-like, shape=[..., n, n]
            Image point.

        Returns
        -------
        base_point : array-like, shape=[..., n, n]
            Base point.
        """
        return expmh(self._unique_diag_mat(image_point) + image_point)

    def tangent_diffeomorphism(self, tangent_vec, base_point=None, image_point=None):
        r"""Tangent diffeomorphism at base point.

        df_p is a linear map from T_pM to T_f(p)N.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.
        image_point : array-like, shape=[..., n, n]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., n, n]
            Image tangent vector at image of the base point.
        """
        if base_point is None:
            base_point = self.inverse_diffeomorphism(image_point)

        return off_map(
            SymMatrixLog.tangent_diffeomorphism(
                tangent_vec=tangent_vec, base_point=base_point
            )
        )

    def _divided_difference_exp(self, eigvals):
        r"""
        First divided difference function of the exponential, exp^(1).

        exp^(1) = (exp(x)-exp(y))/(x-y) if x!=y, exp'(x)=exp(x) if x=y.

        Parameters
        ----------
        eigvals : array-like, shape=[n]
            Typically eigenvalues of the matrix.

        Returns
        -------
        divided_diffs : array-like, shape=[..., n, n]
            First divided difference function of the exponential.

        """
        eigvals_ = gs.expand_dims(eigvals, axis=-2)
        eigvals_t = gs.expand_dims(eigvals, axis=-1)

        eigvals_diff = eigvals_ - eigvals_t

        mask = gs.logical_and(-gs.atol < eigvals_diff, eigvals_diff < gs.atol)

        exp_eigvals = gs.exp(eigvals)

        exp_eigvals_ = gs.expand_dims(exp_eigvals, axis=-2)
        exp_eigvals_t = gs.expand_dims(exp_eigvals, axis=-1)
        default_vals = exp_eigvals_ - gs.zeros((eigvals.shape[-1], 1))

        return gs.where(
            mask,
            default_vals,
            gs.divide(exp_eigvals_ - exp_eigvals_t, eigvals_diff, ignore_div_zero=True),
        )

    def _build_H_0_matrix(self, image_base_point=None, base_point=None):
        r"""
        Builds the SPD matrix H_0 where each coefficient is 
        (H_0)_il = \sum_{j,k} P_ij*P_ik*P_lj*P_lk*exp^(1)(d_j, d_k)
        where PDP^t = _unique_diag_mat(S)+S, S hollow matrix. Used to
        compute the pushforward of the _unique_diag_mat application.

        Parameters
        ----------
        image_base_point : array-like, shape=[..., n, n]
            Image of base point by the diffeomorphism.
        base_point : array-like, optional
            Base point.

        Returns
        -------
        H_0 : array-like, shape=[..., n, n]
            matrix H_0.
        mat : array-like, shape=[..., n, n]
            Matrix such that its exponential is Cor^+.

        """
        n = self.space.n
        if base_point is None:
            sym_mat = image_base_point
            mat = sym_mat + self._unique_diag_mat(sym_mat)
        else:
            mat = logmh(base_point)

        eigvals, eigvecs = gs.linalg.eigh(mat)

        H_0 = gs.zeros(mat.shape[:-2] + (self.space.n, self.space.n))

        divided_diffs = self._divided_difference_exp(eigvals)

        for index_i in range(n):
            for index_j in range(n):
                val = 0
                for index_k in range(n):
                    for index_l in range(n):
                        val += (
                            eigvecs[..., index_i, index_k]
                            * eigvecs[..., index_j, index_k]
                            * eigvecs[..., index_i, index_l]
                            * eigvecs[..., index_j, index_l]
                            * divided_diffs[..., index_k, index_l]
                        )
                H_0[..., index_i, index_j] = val

        return H_0, mat

    def _tangent_diag_map(
        self, image_tangent_vec, image_base_point=None, base_point=None
    ):
        r"""Tangent _unique_diag_mat at base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.
        image_point : array-like, shape=[..., n, n]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., n, n]
            Image tangent vector at image of the base point.
        mat : array-like, shape=[..., n, n]
            Matrix such that its exponential is Cor^+.
        """
        H_0, mat = self._build_H_0_matrix(
            image_base_point=image_base_point, base_point=base_point
        )
        e = gs.ones(self.space.n)
        vec = gs.matvec(
            gs.linalg.inv(H_0),
            gs.matvec(
                Matrices.to_diagonal(
                    SymMatrixLog.inverse_tangent_diffeomorphism(
                        image_point=mat, image_tangent_vec=image_tangent_vec
                    )
                ),
                e,
            ),
        )
        return gs.vec_to_diag(-vec), mat

    def inverse_tangent_diffeomorphism(
        self, image_tangent_vec, image_point=None, base_point=None
    ):
        r"""Inverse tangent diffeomorphism at image point.

        df^-1_p is a linear map from T_f(p)N to T_pM

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., n, n]
            Image tangent vector at image point.
        image_point : array-like, shape=[..., n, n]
            Image point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        """
        diff_D, sym_mat = self._tangent_diag_map(
            image_base_point=image_point,
            base_point=base_point,
            image_tangent_vec=image_tangent_vec,
        )
        return SymMatrixLog.inverse_tangent_diffeomorphism(
            image_point=sym_mat, image_tangent_vec=image_tangent_vec + diff_D
        )


class OffLogMetric(PullbackDiffeoMetric):
    """Pullback metric via a diffeomorphism.

    Diffeormorphism between full-rank correlation matrices and
    hollow matrices endowed with a permutation-invariant metric.

    For more details, check section 8.2.2 [T2022]_.
    """

    def __init__(self, space, alpha=1.0, beta=1.0, gamma=1.0):
        diffeo = OffLogDiffeo(space)

        image_space = SymmetricHollowMatrices(n=space.n, equip=False).equip_with_metric(
            HollowMatricesPermutationInvariantMetric,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        super().__init__(space=space, diffeo=diffeo, image_space=image_space)
