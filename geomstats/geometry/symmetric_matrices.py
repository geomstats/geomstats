"""The vector space of symmetric matrices."""

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.matrices import Matrices

EPSILON = 1e-6
TOLERANCE = 1e-12


class SymmetricMatrices(EmbeddedManifold):
    """Class for the vector space of symmetric matrices of size n."""

    def __init__(self, n):
        assert isinstance(n, int) and n > 0
        super(SymmetricMatrices, self).__init__(
            dimension=int(n * (n + 1) / 2),
            embedding_manifold=Matrices(n, n))
        self.n = n

    def belongs(self, mat, atol=TOLERANCE):
        """Check if mat belongs to the vector space of symmetric matrices."""
        return Matrices.is_symmetric(mat=mat, atol=atol)

    @staticmethod
    def vector_from_symmetric_matrix(mat):
        """Convert the symmetric part of a symmetric matrix into a vector."""
        mat = gs.to_ndarray(mat, to_ndim=3)
        assert gs.all(Matrices.is_symmetric(mat))
        mat = Matrices.make_symmetric(mat)
        _, dim, _ = mat.shape
        i, j = gs.tril_indices(dim)
        return mat[:, i, j]

    @staticmethod
    def symmetric_matrix_from_vector(vec):
        """Convert a vector into a symmetric matrix."""
        vec = gs.to_ndarray(vec, to_ndim=2)
        n_samples, vec_dim = vec.shape
        mat_dim = (gs.sqrt(8. * vec_dim + 1) - 1) / 2
        if mat_dim != int(mat_dim):
            raise ValueError('Invalid input dimension, it must be of the form'
                             '(n_samples, n * (n - 1) / 2)')
        mat_dim = int(mat_dim)
        mask = gs.tril(gs.ones((mat_dim, mat_dim))) != 0
        sym = gs.zeros((n_samples, mat_dim, mat_dim))
        sym[..., mask != 0] = vec
        sym.swapaxes(-1, -2)[..., mask] = vec
        return Matrices.make_symmetric(sym)

    def expm(self, x):
        """
        Compute the matrix exponential.

        Parameters
        ----------
        x : array_like, shape=[n_samples, n, n]
            Symmetric matrix.

        Returns
        -------
        exponential : array_like, shape=[n_samples, n, n]
            Exponential of x.
        """
        eigvals, eigvecs = gs.linalg.eigh(x)
        eigvals = gs.exp(eigvals)
        eigvals = gs.from_vector_to_diagonal_matrix(eigvals)
        transp_eigvecs = gs.transpose(eigvecs, axes=(0, 2, 1))
        exponential = gs.matmul(eigvecs, eigvals)
        exponential = gs.matmul(exponential, transp_eigvecs)
        return exponential
