"""The manifold of Positive Semi Definite matrices of rank k."""

import math

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricBuresWasserstein, SPDMetricAffine, SPDMetricEuclidean, \
    SPDMetricLogEuclidean
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


class RankKPSDMatrices(Manifold):
    """Class for the manifold of symmetric positive definite (PSD) matrices.

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
            metric=None,
            default_point_type="matrix",
            default_coords_type="intrinsic",
            **kwargs
    ):
        super(Manifold, self).__init__(**kwargs)
        self.n = n
        self.dim = (int(n * (n + 1) / 2),)
        self.default_point_type = default_point_type
        self.default_coords_type = default_coords_type
        self.metric = metric
        self.rank = k
        self.sym = SymmetricMatrices(self.n)

    def belongs(self, mat, atol=gs.atol):
        """Check if a matrix is symmetric with positive eigenvalues and
         with n-k zero eigenvalues.

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
        """Project a matrix to the space of PSD matrices of rank k.

        First the symmetric part of point is computed, then the eigenvalues
        are floored to zeros. To ensure rank k, n-k eigenvalues are set to 0

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to project.

        Returns
        -------
        projected: array-like, shape=[..., n, n]
            PSD matrix.
        """

        sym = Matrices(self.n, self.n).to_symmetric(point)
        eigvals, eigvecs = gs.linalg.eigh(sym)
        regularized = gs.where(eigvals < 0, 0, eigvals)
        regularized[0: (self.n - self.rank)] = [0] * (self.n - self.rank)
        reconstruction = gs.einsum("...ij,...j->...ij", eigvecs, regularized)
        return Matrices.mul(reconstruction, Matrices.transpose(eigvecs))
        # ANNA - how can we handle this case?
        # the rank is lower because there are more than n-k zeros eigenvalues

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in PSD(n,k) from the log-uniform distribution of SPD matrices
        and adding zero eigenvalues.

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
        if n_samples > 1:
            psd_mat = [self.projection(i) for i in spd_mat]
        else:
            psd_mat = [self.projection(spd_mat)]
        return psd_mat

    # ANNA add the correct citation of Yann's work

    def is_tangent(self, vector, base_point):
        """Check if the vector belongs to the tangent space at the input point.

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
            Boolean denoting if vector belongs to tangent space at base_point.
        References
        ----------
        .. [TP2019] Thanwerdas, Pennec. "Is affine-invariance well defined on
          SPD matrices? A principled continuum of metrics" Proc. of GSI, 2019.
          https://arxiv.org/abs/1906.01349
        """

        vector_sym = [
            vector if self.sym.belongs(vector) else self.sym.projection(vector)
        ][0]
        # check if symmetric
        r, delta, rt = gs.linalg.svd(base_point)
        rort = r[:, self.n - self.rank: self.n]
        rort_t = rt[self.n - self.rank: self.n, :]
        check = gs.matmul(
            gs.matmul(gs.matmul(rort, rort_t), vector_sym), gs.matmul(rort, rort_t)
        )
        if (
                gs.logical_and(
                    gs.less_equal(check, -gs.atol), gs.greater(check, gs.atol)
                ).sum()
                == 0
        ):
            return True
        else:
            return False

    def to_tangent(self, vector, base_point):
        """Project the input vector to the tangent space of PSD(n,k) at base_point.

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
        if self.is_tangent(vector, base_point):
            return vector
        else:
            vector_sym = [
                vector if self.sym.belongs(vector) else self.sym.projection(vector)
            ][0]
            r, delta, rt = gs.linalg.svd(base_point)
            rort = r[:, self.n - self.rank: self.n]
            rort_t = rt[self.n - self.rank: self.n, :]
            return (
                    gs.matmul(
                        gs.matmul(gs.matmul(rort, rort_t), vector_sym),
                        gs.matmul(rort, rort_t),
                    )
                    + vector_sym
            )


PSDMetricBuresWasserstein = SPDMetricBuresWasserstein

PSDMetricEuclidean = SPDMetricEuclidean

PSDMetricLogEuclidean = SPDMetricLogEuclidean

PSDMetricAffine = SPDMetricAffine


class PSDMatrices(RankKPSDMatrices, SPDMatrices):
    r"""Class for the psd matrices. The class is recirecting to the correct embedding manifold.
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

    def __new__(cls, n, k=None, metric=None, default_point_type="matrix", default_coords_type="intrinsic", ):
        if k == None:
            raise NotImplementedError(
                "PSD matrices of all ranks is not ready implemented"
            )
        elif n > k:
            return RankKPSDMatrices(n, k)
        elif n == k:
            return SPDMatrices(n)
