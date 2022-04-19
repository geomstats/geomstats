r"""The manifold of Positive Semi Definite matrices of rank k PSD(n,k).

Lead author: Anna Calissano.
"""

import geomstats.backend as gs
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.full_rank_matrices import FullRankMatrices
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


class RankKPSDMatrices(Manifold):
    r"""Class for PSD(n,k).

    The manifold of Positive Semi Definite matrices of rank k: PSD(n,k).

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    k: int
        Integer representing the rank of the matrix (k<n).
    """

    def __init__(self, n, k, **kwargs):
        super(RankKPSDMatrices, self).__init__(
            **kwargs,
            dim=int(k * n - k * (k + 1) / 2),
            shape=(n, n),
        )
        self.n = n
        self.rank = k
        self.sym = SymmetricMatrices(self.n)

    def belongs(self, mat, atol=gs.atol):
        r"""Check if the matrix belongs to the space.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix to be checked.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if mat is an SPD matrix.
        """
        is_symmetric = self.sym.belongs(mat, atol)
        eigvalues = gs.linalg.eigvalsh(mat)
        is_semipositive = gs.all(eigvalues > -atol, axis=-1)
        is_rankk = gs.sum(gs.where(eigvalues < atol, 0, 1), axis=-1) == self.rank
        belongs = gs.logical_and(
            gs.logical_and(is_symmetric, is_semipositive), is_rankk
        )
        return belongs

    def projection(self, point):
        r"""Project a matrix to the space of PSD matrices of rank k.

        The nearest symmetric positive semidefinite matrix in the
        Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
        where H is the symmetric polar factor of B=(A + A')/2.
        As [Higham1988] is turning the matrix into a PSD, the rank
        is then forced to be k.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to project.

        Returns
        -------
        projected: array-like, shape=[..., n, n]
            PSD matrix rank k.

        References
        ----------
        [Higham1988]_    Highamm, N. J.
                        “Computing a nearest symmetric positive semidefinite matrix.”
                        Linear Algebra and Its Applications 103 (May 1, 1988):
                        103-118. https://doi.org/10.1016/0024-3795(88)90223-6
        """
        sym = Matrices.to_symmetric(point)
        _, s, v = gs.linalg.svd(sym)
        h = gs.matmul(Matrices.transpose(v), s[..., None] * v)
        sym_proj = (sym + h) / 2
        eigvals, eigvecs = gs.linalg.eigh(sym_proj)
        i = gs.array([0] * (self.n - self.rank) + [2 * gs.atol] * self.rank)
        regularized = (
            gs.assignment(eigvals, 0, gs.arange((self.n - self.rank)), axis=0) + i
        )
        reconstruction = gs.einsum("...ij,...j->...ij", eigvecs, regularized)

        return Matrices.mul(reconstruction, Matrices.transpose(eigvecs))

    def random_point(self, n_samples=1, bound=1.0):
        r"""Sample in PSD(n,k) from the log-uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample in the tangent space.
            Optional, default: 1.

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

    def is_tangent(self, vector, base_point, tangent_atol=gs.atol):
        r"""Check if the vector belongs to the tangent space.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Matrix to check if it belongs to the tangent space.
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.
            Optional, default: None.
        tangent_atol: float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if vector belongs to tangent space
            at base_point.
        """
        vector_sym = Matrices(self.n, self.n).to_symmetric(vector)

        _, r = gs.linalg.eigh(base_point)
        r_ort = r[..., :, self.n - self.rank : self.n]
        r_ort_t = Matrices.transpose(r_ort)
        rr = gs.matmul(r_ort, r_ort_t)
        candidates = Matrices.mul(rr, vector_sym, rr)
        result = gs.all(gs.isclose(candidates, 0.0, tangent_atol), axis=(-2, -1))
        return result

    def to_tangent(self, vector, base_point):
        r"""Project to the tangent space of PSD(n,k) at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Matrix to check if it belongs to the tangent space.
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.
            Optional, default: None.

        Returns
        -------
        tangent : array-like, shape=[...,n,n]
            Projection of the tangent vector at base_point.
        """
        vector_sym = Matrices(self.n, self.n).to_symmetric(vector)
        _, r = gs.linalg.eigh(base_point)
        r_ort = r[..., :, self.n - self.rank : self.n]
        r_ort_t = Matrices.transpose(r_ort)
        rr = gs.matmul(r_ort, r_ort_t)
        return vector_sym - Matrices.mul(rr, vector_sym, rr)


PSDMetricEuclidean = SPDMetricEuclidean


class PSDMatrices(RankKPSDMatrices, SPDMatrices):
    r"""Class for the psd matrices.

    The class is redirecting to the correct embedding manifold.
    The stratum PSD rank k if the matrix is not full rank.
    The top stratum SPD if the matrix is full rank.
    The whole stratified space of PSD if no rank is specified.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices : n x n.
    k : int
        Integer representing the rank of the matrices.
    """

    def __new__(
        cls,
        n,
        k,
        **kwargs,
    ):
        """Instantiate class from one of the parent classes."""
        if n > k:
            return RankKPSDMatrices(n, k, **kwargs)
        if n == k:
            return SPDMatrices(n, **kwargs)
        raise NotImplementedError("The PSD matrices is not implemented yet.")


class BuresWassersteinBundle(FullRankMatrices, FiberBundle):
    """Class for the quotient structure on PSD matrices."""

    def __init__(self, n, k):
        super(BuresWassersteinBundle, self).__init__(
            n=n,
            k=k,
            base=PSDMatrices(n, k),
            group=SpecialOrthogonal(k),
            ambient_metric=MatricesMetric(n, k),
        )

    @staticmethod
    def riemannian_submersion(point):
        """Project."""
        return Matrices.mul(point, Matrices.transpose(point))

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        """Differential."""
        product = Matrices.mul(base_point, Matrices.transpose(tangent_vec))
        return 2 * Matrices.to_symmetric(product)

    def lift(self, point):
        """Find a representer in top space."""
        eigvals, eigvecs = gs.linalg.eigh(point)
        return gs.einsum(
            "...ij,...j->...ij", eigvecs[..., -self.k :], eigvals[..., -self.k :] ** 0.5
        )

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        """Horizontal lift of a tangent vector."""
        if fiber_point is None:
            fiber_point = self.lift(base_point)
        transposed_point = Matrices.transpose(fiber_point)
        alignment = Matrices.mul(transposed_point, fiber_point)
        projector = Matrices.mul(fiber_point, GeneralLinear.inverse(alignment))
        right_term = Matrices.mul(transposed_point, tangent_vec, fiber_point)
        sylvester = gs.linalg.solve_sylvester(alignment, alignment, right_term)
        skew_term = Matrices.mul(projector, sylvester)
        orth_proj = gs.eye(self.n) - Matrices.mul(projector, transposed_point)
        orth_part = Matrices.mul(orth_proj, tangent_vec, projector)
        return skew_term + orth_part

    def vertical_projection(self, tangent_vec, base_point, return_skew=False):
        r"""Project to vertical subspace.

        Compute the vertical component of a tangent vector :math:`w` at a
        base point :math:`x` by solving the sylvester equation:
        .. math::
                        `Axx^T + xx^TA = wx^T - xw^T`

        where A is skew-symmetric. Then Ax is the vertical projection of w.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, k]
            Tangent vector to the pre-shape space at `base_point`.
        base_point : array-like, shape=[..., n, k]
            Point on the pre-shape space.
        return_skew : bool
            Whether to return the skew-symmetric matrix A.
            Optional, default: False

        Returns
        -------
        vertical : array-like, shape=[..., n, k]
            Vertical component of `tangent_vec`.
        skew : array-like, shape=[..., m_ambient, m_ambient]
            Vertical component of `tangent_vec`.
        """
        transposed_point = Matrices.transpose(base_point)
        left_term = gs.matmul(transposed_point, base_point)
        alignment = gs.matmul(Matrices.transpose(tangent_vec), base_point)
        right_term = alignment - Matrices.transpose(alignment)
        skew = gs.linalg.solve_sylvester(left_term, left_term, right_term)

        vertical = -gs.matmul(base_point, skew)
        return (vertical, skew) if return_skew else vertical

    def align(self, point, base_point, **kwargs):
        """Align point to base_point.

        Find the optimal rotation R in SO(m) such that the base point and
        R.point are well positioned.

        Parameters
        ----------
        point : array-like, shape=[..., n, k]
            Point on the manifold.
        base_point : array-like, shape=[..., n, k]
            Point on the manifold.

        Returns
        -------
        aligned : array-like, shape=[..., n, k]
            R.point.
        """
        return Matrices.align_matrices(point, base_point)


class PSDMetricBuresWasserstein(QuotientMetric):
    """Bures-Wasserstein metric for fixed rank PSD matrices."""

    def __init__(self, n, k):
        fiber_bundle = BuresWassersteinBundle(n, k)
        super(PSDMetricBuresWasserstein, self).__init__(
            fiber_bundle=fiber_bundle, shape=(n, k)
        )
