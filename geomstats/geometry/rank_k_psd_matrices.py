r"""The manifold of Positive Semi Definite matrices of rank k PSD(n,k).

Lead author: Anna Calissano.
"""

import geomstats.backend as gs
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.full_rank_matrices import FullRankMatrices
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.group_action import SpecialOrthogonalComposeAction
from geomstats.geometry.manifold import Manifold, register_quotient
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDEuclideanMetric, SPDMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


class RankKPSDMatrices(Manifold):
    r"""Class for PSD(n,k).

    The manifold of positive semi-definite matrices of rank k: PSD(n,k).

    Parameters
    ----------
    n : int
        Size of the matrices: n x n.
    k : int
        Rank of the matrices. Must be lower than `n`.
    equip : bool
        If True, equip the space with its default metric.
    """

    def __init__(self, n, k, equip=True):
        if not k < n:
            raise ValueError("k must be lower than n")

        super().__init__(
            dim=int(k * n - k * (k + 1) / 2),
            shape=(n, n),
            equip=equip,
            intrinsic=False,
        )
        self.n = n
        self.rank = k
        self.sym = SymmetricMatrices(self.n, equip=False)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return PSDEuclideanMetric

    def belongs(self, point, atol=gs.atol):
        r"""Check if a matrix belongs to the space.

        A matrix belongs to PSD(n,k) if it is symmetric, positive
        semi-definite and has exactly k eigenvalues larger than `atol`.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to check.
        atol : float
            Absolute tolerance.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if `point` is a PSD matrix of rank k.
        """
        shape_belongs = self.shape == point.shape[-self.point_ndim :]
        if not shape_belongs:
            shape = point.shape[: -self.point_ndim]
            return gs.zeros(shape, dtype=bool)

        is_symmetric = self.sym.belongs(point, atol)
        eigvalues = gs.linalg.eigvalsh(point)
        is_semipositive = gs.all(eigvalues > -atol, axis=-1)
        sum_eig = gs.sum(gs.where(eigvalues < atol, 0, 1), axis=-1)
        is_rankk = gs.isclose(sum_eig, self.rank, atol=atol)
        return gs.logical_and(gs.logical_and(is_symmetric, is_semipositive), is_rankk)

    def projection(self, point):
        r"""Project a matrix to the space of PSD matrices of rank k.

        The nearest symmetric positive semi-definite matrix in the
        Frobenius norm to an arbitrary real matrix :math:`A` is
        :math:`(B + H) / 2`, where :math:`H` is the symmetric polar factor of
        :math:`B = (A + A^T) / 2` [Higham1988]_. The rank is then forced to be
        k by keeping the k largest eigenvalues and setting the others to zero.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to project.

        Returns
        -------
        projected : array-like, shape=[..., n, n]
            PSD matrix of rank k.

        Notes
        -----
        If :math:`B` has fewer than k positive eigenvalues, the nearest rank-k
        PSD matrix does not exist. The k largest eigenvalues are shifted by
        `2 * gs.atol` so that the result still has rank k.

        References
        ----------
        .. [Higham1988] Higham, N. J.
            “Computing a nearest symmetric positive semidefinite matrix.”
            Linear Algebra and Its Applications 103 (May 1, 1988):
            103-118. https://doi.org/10.1016/0024-3795(88)90223-6
        """
        sym = Matrices.to_symmetric(point)
        _, s, v = gs.linalg.svd(sym)
        h = gs.matmul(gs.transpose(v), s[..., None] * v)
        sym_proj = (sym + h) / 2
        eigvals, eigvecs = gs.linalg.eigh(sym_proj)
        k = self.n - self.rank
        i = gs.array([0.0] * k + [2 * gs.atol] * self.rank)
        regularized = (
            gs.hstack([gs.zeros(point.shape[:-2] + (k,)), eigvals[..., k:]]) + i
        )
        reconstruction = gs.einsum("...ij,...j->...ij", eigvecs, regularized)

        return Matrices.mul(reconstruction, gs.transpose(eigvecs))

    def random_point(self, n_samples=1, bound=1.0):
        r"""Sample random points in PSD(n,k).

        Samples SPD matrices from the log-uniform distribution and projects
        them to PSD(n,k).

        Parameters
        ----------
        n_samples : int
            Number of samples.
        bound : float
            Bound of the interval in which to sample the symmetric matrices
            whose matrix exponential is projected.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled in PSD(n,k).
        """
        n = self.n
        size = (n_samples, n, n) if n_samples != 1 else (n, n)
        mat = bound * (2 * gs.random.rand(*size) - 1)
        spd_mat = GeneralLinear.exp(Matrices.to_symmetric(mat))
        return self.projection(spd_mat)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        r"""Check if a vector belongs to the tangent space at a base point.

        A symmetric matrix :math:`V` is tangent at :math:`P` if
        :math:`U_\perp^T V U_\perp = 0`, where the columns of
        :math:`U_\perp` span the null space of :math:`P`.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Matrix to check.
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.
        atol : float
            Absolute tolerance.

        Returns
        -------
        is_tangent : array-like, shape=[...,]
            Boolean denoting if `vector` belongs to the tangent space
            at `base_point`.
        """
        vector_sym = Matrices.to_symmetric(vector)

        r, _, _ = gs.linalg.svd(base_point)
        r_ort = r[..., :, -(self.n - self.rank) :]
        r_ort_t = gs.transpose(r_ort)
        rr = gs.matmul(r_ort, r_ort_t)
        candidates = Matrices.mul(rr, vector_sym, rr)
        return gs.all(gs.isclose(candidates, 0.0, atol=atol), axis=(-2, -1))

    def to_tangent(self, vector, base_point):
        r"""Project a vector to the tangent space at a base point.

        The symmetric part :math:`V` of `vector` is projected to
        :math:`V - R V R`, where :math:`R = U_\perp U_\perp^T` is the
        orthogonal projector onto the null space of the base point.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Matrix to project.
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        """
        vector_sym = Matrices.to_symmetric(vector)
        r, _, _ = gs.linalg.svd(base_point)
        r_ort = r[..., :, self.rank :]
        r_ort_t = gs.transpose(r_ort)
        rr = gs.matmul(r_ort, r_ort_t)
        return vector_sym - Matrices.mul(rr, vector_sym, rr)


PSDEuclideanMetric = SPDEuclideanMetric


class PSDMatrices:
    r"""Class for the PSD matrices of a given rank.

    Instantiates the stratum of PSD matrices of rank `k`:
    `RankKPSDMatrices` if `k < n`, and `SPDMatrices` (the top stratum)
    if `k == n`.

    Parameters
    ----------
    n : int
        Size of the matrices: n x n.
    k : int
        Rank of the matrices.
    equip : bool
        If True, equip the space with its default metric.
    """

    def __new__(
        cls,
        n,
        k,
        equip=True,
    ):
        """Instantiate the stratum of PSD matrices of rank `k`."""
        if n > k:
            return RankKPSDMatrices(n, k, equip=equip)
        if n == k:
            return SPDMatrices(n, equip=equip)
        raise NotImplementedError("The PSD matrices is not implemented yet.")


class BuresWassersteinBundle(FiberBundle):
    r"""Bures-Wasserstein fiber bundle over PSD matrices.

    The total space is the space of full-rank n x k matrices with the
    Euclidean metric, and the submersion is :math:`X \mapsto X X^T`.
    The fiber over :math:`X X^T` is :math:`X O(k)`.

    Parameters
    ----------
    total_space : FullRankMatrices
        Space of full-rank n x k matrices.
    """

    def __init__(self, total_space):
        super().__init__(total_space=total_space)

    @staticmethod
    def riemannian_submersion(point):
        r"""Project a point of the total space to the base space.

        Parameters
        ----------
        point : array-like, shape=[..., n, k]
            Point on the total space.

        Returns
        -------
        projected_point : array-like, shape=[..., n, n]
            PSD matrix :math:`X X^T`.
        """
        return Matrices.mul(point, gs.transpose(point))

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        r"""Compute the differential of the submersion.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, k]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., n, k]
            Point on the total space.

        Returns
        -------
        projected_tangent_vec : array-like, shape=[..., n, n]
            Tangent vector :math:`X W^T + W X^T` to the base space.
        """
        product = Matrices.mul(base_point, gs.transpose(tangent_vec))
        return product + gs.transpose(product)

    def lift(self, point):
        r"""Find a point of the total space in the fiber of a point.

        Uses the eigendecomposition :math:`P = U \Lambda U^T`, and returns
        :math:`U_k \Lambda_k^{1/2}`, where :math:`U_k` and :math:`\Lambda_k`
        correspond to the k largest eigenvalues.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point on the base space.

        Returns
        -------
        lift : array-like, shape=[..., n, k]
            Point on the total space.
        """
        k = self._total_space.k
        eigvals, eigvecs = gs.linalg.eigh(point)
        return gs.einsum(
            "...ij,...j->...ij", eigvecs[..., -k:], eigvals[..., -k:] ** 0.5
        )

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        """Compute the horizontal lift of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector to the base space at `base_point`.
        base_point : array-like, shape=[..., n, n]
            Point on the base space. Used to compute `fiber_point` if it is
            not given.
        fiber_point : array-like, shape=[..., n, k]
            Point on the total space in the fiber of `base_point`.

        Returns
        -------
        horizontal_vec : array-like, shape=[..., n, k]
            Horizontal tangent vector to the total space at `fiber_point`.
        """
        n = self._total_space.n
        if fiber_point is None:
            fiber_point = self.lift(base_point)
        transposed_point = gs.transpose(fiber_point)
        alignment = Matrices.mul(transposed_point, fiber_point)
        projector = Matrices.mul(fiber_point, GeneralLinear.inverse(alignment))
        right_term = Matrices.mul(transposed_point, tangent_vec, fiber_point)
        sylvester = gs.linalg.solve_sylvester(alignment, alignment, right_term)
        skew_term = Matrices.mul(projector, sylvester)
        orth_proj = gs.eye(n) - Matrices.mul(projector, transposed_point)
        orth_part = Matrices.mul(orth_proj, tangent_vec, projector)
        return skew_term + orth_part

    def vertical_projection(self, tangent_vec, base_point, return_skew=False):
        r"""Project to vertical subspace.

        Compute the vertical component of a tangent vector :math:`W` at a
        point :math:`X` by solving the Sylvester equation

        .. math::
            X^T X A + A X^T X = W^T X - X^T W

        for a skew-symmetric :math:`A`. The vertical component is then
        :math:`-X A`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, k]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., n, k]
            Point on the total space.
        return_skew : bool
            Whether to return the skew-symmetric matrix :math:`A`.

        Returns
        -------
        vertical : array-like, shape=[..., n, k]
            Vertical component of `tangent_vec`.
        skew : array-like, shape=[..., k, k]
            Skew-symmetric matrix :math:`A`. Only returned if `return_skew`
            is True.
        """
        transposed_point = gs.transpose(base_point)
        left_term = gs.matmul(transposed_point, base_point)
        alignment = gs.matmul(gs.transpose(tangent_vec), base_point)
        right_term = alignment - gs.transpose(alignment)
        skew = gs.linalg.solve_sylvester(left_term, left_term, right_term)

        vertical = -gs.matmul(base_point, skew)
        return (vertical, skew) if return_skew else vertical

    def align(self, point, base_point):
        r"""Align a point to a base point.

        Finds the orthogonal matrix :math:`R \in O(k)` minimizing
        :math:`\|X R - Y\|`, where :math:`X` is `point` and :math:`Y` is
        `base_point`. The alignment is over :math:`O(k)`, not :math:`SO(k)`,
        since the fiber over :math:`X X^T` is :math:`X O(k)`.

        Parameters
        ----------
        point : array-like, shape=[..., n, k]
            Point on the total space.
        base_point : array-like, shape=[..., n, k]
            Point on the total space.

        Returns
        -------
        aligned : array-like, shape=[..., n, k]
            Aligned point :math:`X R`.
        """
        return Matrices.align_matrices(point, base_point, flip=False)


class PSDBuresWassersteinMetric(QuotientMetric):
    r"""Bures-Wasserstein metric on PSD matrices of fixed rank.

    Quotient metric of the Euclidean metric on full-rank n x k matrices,
    through the submersion :math:`X \mapsto X X^T`.

    Parameters
    ----------
    space : RankKPSDMatrices or SPDMatrices
        Space of PSD matrices of rank k.
    total_space : FullRankMatrices
        Space of full-rank n x k matrices. If None, it is created and
        equipped with the Euclidean metric.
    """

    def __init__(self, space, total_space=None):
        if total_space is None:
            k = space.rank if hasattr(space, "rank") else space.n
            total_space = FullRankMatrices(space.n, k, equip=False)
            total_space.equip_with_metric(MatricesMetric)

        if not hasattr(total_space, "group_action"):
            total_space.equip_with_group_action(
                SpecialOrthogonalComposeAction(total_space.k)
            )

        if not hasattr(total_space, "quotient"):
            total_space.equip_with_quotient()

        super().__init__(space=space, total_space=total_space)


register_quotient(
    Space=FullRankMatrices,
    Metric=MatricesMetric,
    GroupAction=SpecialOrthogonalComposeAction,
    FiberBundle=BuresWassersteinBundle,
)
