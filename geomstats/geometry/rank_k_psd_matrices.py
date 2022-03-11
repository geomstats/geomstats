r"""The manifold of Positive Semi Definite matrices of rank k PSD(n,k).

Lead author: Anna Calissano.
"""

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import (
    SPDMatrices,
    SPDMetricBuresWasserstein,
    SPDMetricEuclidean,
)
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


PSDMetricBuresWasserstein = SPDMetricBuresWasserstein

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
        Integer representing the shapes of the matrices : n x n.
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
