"""The vector space of symmetric matrices."""

import logging

import geomstats.backend as gs
import geomstats.vectorization
from geomstats import algebra_utils
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.matrices import Matrices

EPSILON = 1e-6
TOLERANCE = 1e-12


class SymmetricMatrices(EmbeddedManifold):
    """Class for the vector space of symmetric matrices of size n."""

    def __init__(self, n):
        super(SymmetricMatrices, self).__init__(
            dimension=int(n * (n + 1) / 2),
            embedding_manifold=Matrices(n, n))
        self.n = n

    def belongs(self, mat, atol=TOLERANCE):
        """Check if mat belongs to the vector space of symmetric matrices."""
        return self.embedding_manifold.is_symmetric(mat=mat, atol=atol)

    def get_basis(self):
        """Compute the basis of the vector space of symmetric matrices."""
        basis = [
            gs.array_from_sparse([
                (row, col), (col, row)], [1., 1.], (self.n, self.n))
            for row in gs.arange(self.n)
            for col in gs.arange(row, self.n)]
        basis = gs.stack(
            basis) * (gs.ones((self.n, self.n)) - 1. / 2 * gs.eye(self.n))
        return basis

    basis = property(get_basis)

    @staticmethod
    @geomstats.vectorization.decorator(['matrix'])
    def vector_from_symmetric_matrix(mat):
        """Convert the symmetric part of a symmetric matrix into a vector."""
        if not gs.all(Matrices.is_symmetric(mat)):
            logging.warning('non-symmetric matrix encountered.')
        mat = Matrices.make_symmetric(mat)
        _, dim, _ = mat.shape
        i, j = gs.triu_indices(dim)
        vec = mat[:, i, j]
        return vec

    @staticmethod
    @geomstats.vectorization.decorator(['vector', 'else'])
    def symmetric_matrix_from_vector(vec, dtype=gs.float32):
        """Convert a vector into a symmetric matrix."""
        vec_dim = vec.shape[-1]
        mat_dim = (gs.sqrt(8. * vec_dim + 1) - 1) / 2
        if mat_dim != int(mat_dim):
            raise ValueError('Invalid input dimension, it must be of the form'
                             '(n_samples, n * (n - 1) / 2)')
        mat_dim = int(mat_dim)
        mask = 2 * gs.ones((mat_dim, mat_dim)) - gs.eye(mat_dim)
        indices = list(zip(*gs.triu_indices(3)))
        shape = (mat_dim, mat_dim)
        vec = gs.cast(vec, dtype)
        upper_triangular = gs.stack([
            gs.array_from_sparse(indices, data, shape) for data in vec])
        mat = Matrices.make_symmetric(upper_triangular) * mask
        return mat

    @staticmethod
    @geomstats.vectorization.decorator(['matrix'])
    def expm(x):
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
        eigvals = algebra_utils.from_vector_to_diagonal_matrix(eigvals)
        transp_eigvecs = gs.transpose(eigvecs, axes=(0, 2, 1))
        exponential = gs.matmul(eigvecs, eigvals)
        exponential = gs.matmul(exponential, transp_eigvecs)
        return exponential
