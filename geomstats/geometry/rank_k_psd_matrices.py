r"""The manifold of Positive Semi Definite matrices of rank k PSD(n,k)."""

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import (
    SPDMatrices,
    SPDMetricBuresWasserstein,
    SPDMetricAffine,
    SPDMetricEuclidean,
    SPDMetricLogEuclidean,
)
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


class RankKPSDMatrices(Manifold):
    r"""Class for the manifold of symmetric positive definite (PSD)
    matrices of rank k: PSD(n,k).

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    k: int
        Integer representing the rank of the matrix (k<n).
    """

    def __init__(
        self,
        n,
        k,
        **kwargs
    ):
        super(RankKPSDMatrices, self).__init__(**kwargs)
        self.n = n
        self.dim = int(k * n - k * (k + 1) / 2)
        self.default_point_type = default_point_type
        self.default_coords_type = default_coords_type
        self.metric = metric
        self.rank = k
        self.sym = SymmetricMatrices(self.n)

    def belongs(self, mat, atol=gs.atol):
        r"""Check if a matrix is symmetric with k positive eigenvalues
        and with n-k zero eigenvalues.

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
        is_rankk = gs.linalg.matrix_rank(mat) == self.rank
        belongs = gs.logical_and(
            gs.logical_and(is_symmetric, is_semipositive), is_rankk
        )
        return belongs

    def projection(self, point):
        r"""Project a matrix to the space of PSD matrices of rank k.
        First the symmetric part of point is computed,
        then the matrix is multiplied by the I_\epsilon matrix.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to project.

        Returns
        -------
        projected: array-like, shape=[..., n, n]
            PSD matrix rank k.
        """

        sym = self.sym.projection(point)
        u, s, vh = gs.linalg.svd(sym, full_matrices=False)
        s[..., self.rank: self.n] = 0
        i = [gs.atol] * self.rank + [0] * (self.n - self.rank)
        return gs.matmul(u, (s + i)[..., None] * vh)

    def random_point(self, n_samples=1, bound=1.0):
        r"""Sample in PSD(n,k) from the log-uniform distribution
        of SPD matrices and then projecting onto the space with
        the projection function.

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

    def is_tangent(self, vector, base_point):
        r"""Check if the vector belongs to the tangent space
        at the input point.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Matrix to check if it belongs to the tangent space.
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.
            Optional, default: None.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if vector belongs to tangent space
            at base_point.
        """

        vector_sym = Matrices(self.n, self.n).to_symmetric(vector)

        _, r = gs.linalg.eigh(base_point)
        r_ort = r[..., :, self.n - self.rank: self.n]
        r_ort_t = Matrices.transpose(r_ort)

        candidates = gs.matmul(
            gs.matmul(gs.matmul(r_ort, r_ort_t), vector_sym), gs.matmul(r_ort, r_ort_t)
        )

        result = gs.logical_and(
            gs.less_equal(-gs.atol, candidates), gs.greater(gs.atol, candidates)
        ).max(axis=(-2, -1))

        return result

    def to_tangent(self, vector, base_point):
        r"""Project the input vector to the tangent space of PSD(n,k)
        at base_point.

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
        delta, r = gs.linalg.eigh(base_point)
        r_ort = r[..., :, self.n - self.rank: self.n]
        r_ort_t = Matrices.transpose(r_ort)

        return (vector_sym - gs.matmul(
                gs.matmul(gs.matmul(r_ort, r_ort_t), vector_sym),
                gs.matmul(r_ort, r_ort_t)))


PSDMetricBuresWasserstein = SPDMetricBuresWasserstein

PSDMetricEuclidean = SPDMetricEuclidean

PSDMetricLogEuclidean = SPDMetricLogEuclidean

PSDMetricAffine = SPDMetricAffine


class PSDMatrices(RankKPSDMatrices, SPDMatrices):
    r"""Class for the psd matrices. The class is redirecting to
    the correct embedding manifold.

    The stratum PSD rank k if the matrix is not full rank
    The top stratum SPD if the matrix is full rank
    The whole stratified space of PSD if no rank is specified

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
        if n > k:
            return RankKPSDMatrices(n, k, **kwargs)
        elif n == k:
            return SPDMatrices(n, **kwargs)
