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
from geomstats.geometry.fiber_bundle import DistanceMinimizingAligner, FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.hermitian_matrices import expmh
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.manifold import register_quotient
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
    NullRowSumsPermutationInvariantMetric,
    NullRowSumsSymmetricMatrices,
    SymmetricHollowMatrices,
)
from geomstats.numerics.optimizers import NewtonMethod


def corr_map(point):
    r"""Compute the correlation matrix associated to an SPD matrix.

    .. math::

        \text { Cor }: \Sigma \in \operatorname{Sym}^{+}(n) \longmapsto
        \operatorname{Diag}(\Sigma)^{-1 / 2} \Sigma
        \operatorname{Diag}(\Sigma)^{-1 / 2} \in \operatorname{Cor}^{+}(n)

    Parameters
    ----------
    point : array-like, shape=[..., n, n]
        SPD matrix.

    Returns
    -------
    cor : array_like, shape=[..., n, n]
        Full-rank correlation matrix.
    """
    diagonal = Matrices.diagonal(point) ** (-0.5)
    return FullRankCorrelationMatrices.diag_action(diagonal, point)


def tangent_corr_map(tangent_vec, base_point):
    r"""Compute the differential of the differential of the corr map.

    .. math::

        d_{\Sigma} \operatorname{Cor}(X)=\Delta_{\Sigma}\left[X-\frac{1}{2}
        \left(\Delta_{\Sigma}^2 \operatorname{Diag}(X)
        \Sigma+\Sigma \operatorname{Diag}(X)
        \Delta_{\Sigma}^2\right)\right] \Delta_{\Sigma}

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

    scaled_diag_vec = diagonal_bp ** (-0.5)

    return FullRankCorrelationMatrices.diag_action(scaled_diag_vec, mat)


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
        aligner = (
            DistanceMinimizingAligner(total_space, group_elem_shape=(total_space.n,))
            if gs.has_autodiff()
            else None
        )

        super().__init__(total_space=total_space, aligner=aligner)

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
        return corr_map(point)

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
        return tangent_corr_map(tangent_vec, base_point)

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
        lift = self._total_space.group_action(diagonal_point, tangent_vec)
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

        if not hasattr(total_space, "group_action"):
            total_space.equip_with_group_action(FullRankCorrelationMatrices.diag_action)

        if not hasattr(total_space, "quotient"):
            total_space.equip_with_quotient()

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

    References
    ----------
    .. [T2022] Yann Thanwerdas. Riemannian and stratified
        geometries on covariance and correlation matrices. Differential
        Geometry [math.DG]. Université Côte d'Azur, 2022.
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


class UniqueDiagonalMatrixAlgorithm:
    r"""Find unique diagonal matrix corresponding to a full-rank correlation matrix.

    That is, for all symmetric matrix :math:`S`,
    there exists a unique diagonal matrix :math:`D` such that
    :math:`expm(D+S)` is a full-rank correlation matrix.

    Converges in logarithmic time to the solution of the equation, no closed form.

    Check out Theorem 8.10 of [T2022]_ for more details.

    Parameters
    ----------
    atol : float
        Tolerance to check algorithm convergence.
    max_iter : int
        Maximum iterations.

    References
    ----------
    .. [T2022] Yann Thanwerdas. Riemannian and stratified
        geometries on covariance and correlation matrices. Differential
        Geometry [math.DG]. Université Côte d'Azur, 2022.
    .. [AH2020] Ilya Archakov, and Peter Reinhard Hansen.
        “A New Parametrization of Correlation Matrices.” arXiv, December 3, 2020.
        https://doi.org/10.48550/arXiv.2012.02395.
    """

    def __init__(self, atol=gs.atol, max_iter=100):
        self.atol = atol
        self.max_iter = max_iter

    def _check_convergence(self, new_matrix, matrix):
        mat_diff = new_matrix - matrix
        return gs.linalg.norm(mat_diff, axis=(-2, -1)) < self.atol

    def _call_single(self, sym_mat):
        r"""Find unique diagonal matrix corresponding to a full-rank correlation matrix.

        Parameters
        ----------
        sym_mat : array-like, shape=[n, n]

        Returns
        -------
        diag_mat : array-like, shape=[n, n]
        """
        diag_mat = gs.zeros_like(sym_mat)

        for _ in range(self.max_iter):
            approx_cor_mat = expmh(diag_mat + sym_mat)
            new_diag_mat = diag_mat - logmh(Matrices.to_diagonal(approx_cor_mat))

            if self._check_convergence(new_diag_mat, diag_mat):
                return diag_mat

            diag_mat = new_diag_mat
        else:
            logging.warning(
                "Maximum number of iterations %d reached. The mean may be inaccurate",
                self.max_iter,
            )

        return diag_mat

    def __call__(self, sym_mat):
        r"""Find unique diagonal matrix corresponding to a full-rank correlation matrix.

        Parameters
        ----------
        sym_mat : array-like, shape=[..., n, n]

        Returns
        -------
        diag_mat : array-like, shape=[..., n, n]
        """
        if sym_mat.ndim == 2:
            return self._call_single(sym_mat)

        batch_shape = sym_mat.shape[:-2]
        if len(batch_shape) == 1:
            return gs.stack([self._call_single(sym_mat_) for sym_mat_ in sym_mat])

        mat_shape = sym_mat.shape[-2:]
        flat_sym_mat = gs.reshape(sym_mat, (-1,) + mat_shape)
        out = gs.stack([self._call_single(sym_mat_) for sym_mat_ in flat_sym_mat])
        return gs.reshape(out, batch_shape + mat_shape)


class OffLogDiffeo(Diffeo):
    r"""Off-log diffeomorphism from Cor+ to Hol.

    A diffeomorphism between full-rank correlation matrices Cor+ and
    symmetric hollow matrices Hol:

    .. math::
        \operatorname{Log} = \operatorname{Off} \circ \log :
        \operatorname{Cor}^{+}(n) \longrightarrow \operatorname{Hol}(n)

    Check out chapter 8 of [T2022]_ for more details.

    References
    ----------
    .. [T2022] Yann Thanwerdas. Riemannian and stratified
        geometries on covariance and correlation matrices. Differential
        Geometry [math.DG]. Université Côte d'Azur, 2022.
    """

    def __init__(self):
        super().__init__()
        self.unique_diag_mat = UniqueDiagonalMatrixAlgorithm()

    def __call__(self, base_point):
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

    def inverse(self, image_point):
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
        return expmh(self.unique_diag_mat(image_point) + image_point)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
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
            base_point = self.inverse(image_point)

        return off_map(
            SymMatrixLog.tangent(tangent_vec=tangent_vec, base_point=base_point)
        )

    def _divided_difference_exp(self, eigvals):
        r"""First divided difference function of the exponential, :math:`exp^(1)`.

        If :math:` x \neq y`,

        .. math::

            exp^(1) = (exp(x)-exp(y))/(x-y)

        else:

        .. math::

            exp'(x)=exp(x)

        Parameters
        ----------
        eigvals : array-like, shape=[..., n]
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

    def _build_tangent_diag_aux_mat(self, image_point=None, base_point=None):
        r"""Build auxiliar matrix for tangent diagonal map computation.

        The :math:`H_0` matrix is a SPD matrix where each coefficient is

        .. math::

            (H_0)_il = \sum_{j,k} P_ij*P_ik*P_lj*P_lk*exp^(1)(d_j, d_k)

        where :math:`PDP^t = D+S`, with :math:`D` being a diagonal matrix obtained
        using `UniqueDiagonalMatrixAlgorithm` and :math:`S` is a hollow matrix.

        It is used to compute the pushforward of the
        `UniqueDiagonalMatrixAlgorithm` application.

        Parameters
        ----------
        image_point : array-like, shape=[..., n, n]
            Image of base point by the diffeomorphism.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        h0_mat : array-like, shape=[..., n, n]
            H_0 matrix.
        mat : array-like, shape=[..., n, n]
            Matrix such that its exponential is Cor^+.
        """
        if base_point is None:
            sym_mat = image_point
            mat = sym_mat + self.unique_diag_mat(sym_mat)
        else:
            mat = logmh(base_point)

        eigvals, eigvecs = gs.linalg.eigh(mat)

        h0_mat = gs.zeros(mat.shape)

        divided_diffs = self._divided_difference_exp(eigvals)

        n = h0_mat.shape[-1]
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
                h0_mat[..., index_i, index_j] = val

        return h0_mat, mat

    def _tangent_diag_map(self, image_tangent_vec, image_point=None, base_point=None):
        r"""Tangent unique diagonal matrix at image point.

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
        h0_mat, mat = self._build_tangent_diag_aux_mat(
            image_point=image_point, base_point=base_point
        )
        e = gs.ones(h0_mat.shape[-1])
        vec = gs.matvec(
            gs.linalg.inv(h0_mat),
            gs.matvec(
                Matrices.to_diagonal(
                    SymMatrixLog.inverse_tangent(
                        image_point=mat, image_tangent_vec=image_tangent_vec
                    )
                ),
                e,
            ),
        )
        return gs.vec_to_diag(-vec), mat

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
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
            image_point=image_point,
            base_point=base_point,
            image_tangent_vec=image_tangent_vec,
        )
        return SymMatrixLog.inverse_tangent(
            image_point=sym_mat, image_tangent_vec=image_tangent_vec + diff_D
        )


class OffLogMetric(PullbackDiffeoMetric):
    """Pullback metric via a diffeomorphism.

    Diffeormorphism between full-rank correlation matrices and
    hollow matrices endowed with a permutation-invariant metric.

    For more details, check section 8.2.2 [T2022]_.

    Parameters
    ----------
    space : FullRankCorrelationMatrices
    alpha : float
        Scalar multiplying first term of quadratic form.
    beta : float
        Scalar multiplying second term of quadratic form.
    gamma : float
        Scalar multiplying third term of quadratic form.

    References
    ----------
    .. [T2022] Yann Thanwerdas. Riemannian and stratified
        geometries on covariance and correlation matrices. Differential
        Geometry [math.DG]. Université Côte d'Azur, 2022.
    """

    def __init__(self, space, alpha=None, beta=None, gamma=1.0):
        diffeo = OffLogDiffeo()

        image_space = SymmetricHollowMatrices(n=space.n, equip=False).equip_with_metric(
            HollowMatricesPermutationInvariantMetric,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        super().__init__(space=space, diffeo=diffeo, image_space=image_space)


class SPDScalingFinder:
    r"""Find unique positive diagonal matrix corresponding to an SPD matrix.

    That is, for every symmetric positive-definite matrix :math:`\Sigma`,
    there exists a unique positive diagonal matrix :math:`\Delta` such that
    :math:`\Delta \Sigma \Delta` is a symmetric positive-definite matrix
    with unit row sums.

    This result is known as the existence and uniqueness of the scaling of
    SPD matrices ([T2024]_, [MO1968]_, [JR2009]_).

    It finds the roots of the gradient of the strictly convex map

    .. math::

        F(D) = \frac{1}{2} \mathbb{1}^{\top} D^{\top}
        \Sigma D \mathbb{1}-\operatorname{tr}(\log (D))

    For details, check out [T2024]_'s section 3.5.


    Parameters
    ----------
    root_finder : RootFinder

    References
    ----------
    .. [T2024] Thanwerdas, Yann. “Permutation-Invariant Log-Euclidean Geometries
        on Full-Rank Correlation Matrices.”
        SIAM Journal on Matrix Analysis and Applications, 2024, 930–53.
        https://doi.org/10.1137/22M1538144.
    .. [MO1968] Marshall, Albert W., and Ingram Olkin.
        “Scaling of Matrices to Achieve Specified Row and Column Sums.”
        Numerische Mathematik 12, no. 1 (August 1, 1968): 83–90.
        https://doi.org/10.1007/BF02170999.
    .. [JR2009] Johnson, Charles R., and Robert Reams.
        “Scaling of Symmetric Matrices by Positive Diagonal Congruence.”
        Linear and Multilinear Algebra 57, no. 2 (March 1, 2009): 123–40.
        https://doi.org/10.1080/03081080600872327.
    """

    def __init__(self, root_finder=None):
        if root_finder is None:
            root_finder = NewtonMethod()

        self.root_finder = root_finder

    def _jacobian_f(self, spd_matrix, diag_vec):
        r"""Jacobian of objective function.

        .. math::
            J_{\Sigma}(D) = \Sigma D e - D^{-1} e

        Parameters
        ----------
        spd_matrix : array-like, shape=[..., n, n]
            Symmetric positive-definite matrix.
        diag_vec : array-like, shape=[..., n]
            Vector corresponding to the diagonal of a matrix.

        Returns
        -------
        jacobian : array-like, shape=[..., n]
        """
        return gs.matvec(spd_matrix, diag_vec) - 1.0 / diag_vec

    def _hessian_f(self, spd_matrix, diag_vec):
        r"""Hessian of objective function.

        .. math::
            H_{\Sigma}(D) = \Sigma + D^-2

        Parameters
        ----------
        spd_matrix : array-like, shape=[..., n, n]
            Symmetric positive-definite matrix.
        diag_vec : array-like, shape=[..., n]
            Vector corresponding to the diagonal of a matrix.

        Returns
        -------
        hessian : array-like, shape=[..., n, n]
        """
        return spd_matrix + gs.vec_to_diag(1.0 / diag_vec**2)

    def _call_single(self, spd_matrix):
        """Apply root finder to find scaling.

        Parameters
        ----------
        spd_matrix : array-like, shape=[n, n]
            Symmetric positive-definite matrix.

        Returns
        -------
        diag_vec : array-like, shape=[n]
            Scaling of spd_matrix.
        """
        x0 = gs.ones(spd_matrix.shape[-1])

        func = lambda x: self._jacobian_f(spd_matrix, x)
        jac = lambda x: self._hessian_f(spd_matrix, x)
        res = self.root_finder.root(func, x0, fun_jac=jac)
        return res.x

    def __call__(self, spd_matrix):
        """Apply Newton method to find scaling.

        Parameters
        ----------
        sym_mat : array-like, shape=[..., n, n]
            Symmetric positive-definite matrix.

        Returns
        -------
        diag_vec : array-like, shape=[..., n]
            Scaling of spd_matrix.
        """
        if spd_matrix.ndim == 2:
            return self._call_single(spd_matrix)

        batch_shape = spd_matrix.shape[:-2]
        if len(batch_shape) == 1:
            return gs.stack([self._call_single(sym_mat_) for sym_mat_ in spd_matrix])

        mat_shape = spd_matrix.shape[-2:]
        flat_spd_mat = gs.reshape(spd_matrix, (-1,) + mat_shape)
        out = gs.stack([self._call_single(sym_mat_) for sym_mat_ in flat_spd_mat])
        return gs.reshape(out, batch_shape + (mat_shape.shape[-1],))


class LogScalingDiffeo(Diffeo):
    r"""Off-log diffeomorphism from Cor+ to Row_1^+.

    A diffeomorphism between full-rank correlation matrices :math:`Cor+(n)`
    and the space of symmetric matrices with null row sums :math:`Row_0(n)`.

    .. math::

        \operatorname{Log} ^{\star} =
        \log \left(\mathcal{D}^{\star}(\Sigma)
        \star \Sigma\right): \operatorname{Cor}^{+}(n)
        \longrightarrow \operatorname{Row}_0(n)

    Check out [T2024]_ for more details.

    References
    ----------
    .. [T2024] Thanwerdas, Yann. “Permutation-Invariant Log-Euclidean Geometries
        on Full-Rank Correlation Matrices.”
        SIAM Journal on Matrix Analysis and Applications, 2024, 930–53.
        https://doi.org/10.1137/22M1538144.
    """

    def __init__(self):
        super().__init__()
        self.unique_diag_mat = SPDScalingFinder()

    def __call__(self, base_point):
        """Diffeomorphism at base point.

        NB: congruence action is implictly performed.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., n, n]
            Image point.
        """
        diag_vec = self.unique_diag_mat(base_point)
        unit_row_sum_spd = FullRankCorrelationMatrices.diag_action(diag_vec, base_point)

        return logmh(unit_row_sum_spd)

    def inverse(self, image_point):
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
        return corr_map(expmh(image_point))

    def tangent(self, tangent_vec, base_point=None, image_point=None):
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
        if image_point is None:
            diag_vec = self.unique_diag_mat(base_point)
            base_point_row_1 = FullRankCorrelationMatrices.diag_action(
                diag_vec, base_point
            )
        else:
            base_point_row_1 = expmh(image_point)

        delta = Matrices.diagonal(base_point_row_1) ** 0.5
        aux = FullRankCorrelationMatrices.diag_action(delta, tangent_vec)
        eye = gs.eye(base_point_row_1.shape[-1])
        tangent_vec_row_1_0 = -2 * gs.vec_to_diag(
            gs.sum(
                Matrices.mul(
                    gs.linalg.inv(eye + base_point_row_1),
                    aux,
                ),
                axis=-1,
            )
        )

        tangent_vec_row_1 = aux + 0.5 * (
            Matrices.mul(tangent_vec_row_1_0, base_point_row_1)
            + Matrices.mul(base_point_row_1, tangent_vec_row_1_0)
        )

        return SymMatrixLog.tangent(
            tangent_vec=tangent_vec_row_1,
            base_point=base_point_row_1,
            image_point=image_point,
        )

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
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
        if image_point is not None:
            base_point_row_1 = expmh(image_point)
        else:
            diag_vec = self.unique_diag_mat(base_point)
            base_point_row_1 = FullRankCorrelationMatrices.diag_action(
                diag_vec, base_point
            )

        tangent_vec_row_1 = SymMatrixLog.inverse_tangent(
            image_tangent_vec=image_tangent_vec,
            image_point=image_point,
            base_point=base_point_row_1,
        )
        return tangent_corr_map(tangent_vec_row_1, base_point_row_1)


class LogScaledMetric(PullbackDiffeoMetric):
    """Pullback metric via a diffeomorphism.

    Diffeormorphism between full-rank correlation matrices and
    the space of symmetric matrices with null row sums.

    Check out [T2024]_ for more details.

    Parameters
    ----------
    space : FullRankCorrelationMatrices
    alpha : float
        Scalar multiplying first term of quadratic form.
    delta : float
        Scalar multiplying second term of quadratic form.
    zeta : float
        Scalar multiplying third term of quadratic form.

    References
    ----------
    .. [T2024] Thanwerdas, Yann. “Permutation-Invariant Log-Euclidean Geometries
        on Full-Rank Correlation Matrices.”
        SIAM Journal on Matrix Analysis and Applications, 2024, 930–53.
        https://doi.org/10.1137/22M1538144.
    """

    def __init__(self, space, alpha=None, delta=None, zeta=1.0):
        diffeo = LogScalingDiffeo()

        image_space = NullRowSumsSymmetricMatrices(
            n=space.n, equip=False
        ).equip_with_metric(
            NullRowSumsPermutationInvariantMetric,
            alpha=alpha,
            delta=delta,
            zeta=zeta,
        )

        super().__init__(space=space, diffeo=diffeo, image_space=image_space)


register_quotient(
    Space=SPDMatrices,
    Metric=SPDAffineMetric,
    GroupAction=FullRankCorrelationMatrices.diag_action,
    FiberBundle=CorrelationMatricesBundle,
)
