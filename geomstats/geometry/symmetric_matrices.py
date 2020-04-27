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

    def __init__(self, n, **kwargs):
        super(SymmetricMatrices, self).__init__(
            dim=int(n * (n + 1) / 2),
            embedding_manifold=Matrices(n, n))
        self.n = n

    def belongs(self, mat, atol=TOLERANCE):
        """Check if mat belongs to the vector space of symmetric matrices."""
        check_shape = self.embedding_manifold.belongs(mat)
        return gs.logical_and(check_shape, Matrices.is_symmetric(mat, atol))

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
    def to_vector(mat):
        """Convert the symmetric part of a symmetric matrix into a vector."""
        if not gs.all(Matrices.is_symmetric(mat)):
            logging.warning('non-symmetric matrix encountered.')
        mat = Matrices.to_symmetric(mat)
        _, dim, _ = mat.shape
        indices_i, indices_j = gs.triu_indices(dim)
        vec = []
        for i, j in zip(indices_i, indices_j):
            vec.append(mat[:, i, j])
        vec = gs.stack(vec, axis=1)

        return vec

    @staticmethod
    @geomstats.vectorization.decorator(['vector', 'else'])
    def from_vector(vec, dtype=gs.float32):
        """Convert a vector into a symmetric matrix."""
        vec_dim = vec.shape[-1]
        mat_dim = (gs.sqrt(8. * vec_dim + 1) - 1) / 2
        if mat_dim != int(mat_dim):
            raise ValueError('Invalid input dimension, it must be of the form'
                             '(n_samples, n * (n + 1) / 2)')
        mat_dim = int(mat_dim)
        shape = (mat_dim, mat_dim)
        mask = 2 * gs.ones(shape) - gs.eye(mat_dim)
        indices = list(zip(*gs.triu_indices(mat_dim)))
        vec = gs.cast(vec, dtype)
        upper_triangular = gs.stack([
            gs.array_from_sparse(indices, data, shape) for data in vec])
        mat = Matrices.to_symmetric(upper_triangular) * mask
        return mat

    @classmethod
    @geomstats.vectorization.decorator(['else', 'matrix'])
    def expm(cls, x):
        """
        Compute the matrix exponential for a symmetric matrix.

        Parameters
        ----------
        x : array_like, shape=[..., n, n]
            Symmetric matrix.

        Returns
        -------
        exponential : array_like, shape=[..., n, n]
            Exponential of x.
        """
        return cls.apply_func_to_eigvals(x, gs.exp)

    @classmethod
    def powerm(cls, x, power):
        """
        Compute the matrix power.

        Parameters
        ----------
        x : array_like, shape=[..., n, n]
            Symmetric matrix with non-negative eigenvalues.
        power : float
            The power at which x will be raised.

        Returns
        -------
        powerm : array_like, shape=[..., n, n]
            Matrix power of x.
        """
        def _power(eigvals):
            return gs.power(eigvals, power)
        return cls.apply_func_to_eigvals(x, _power, check_positive=True)

    @staticmethod
    def apply_func_to_eigvals(x, function, check_positive=False):
        """
        Apply function to eigenvalues and reconstruct the matrix.

        Parameters
        ----------
        x : array_like, shape=[..., n, n]
            Symmetric matrix.
        function : callable
            Function to apply to eigenvalues.

        Returns
        -------
        x : array_like, shape=[..., n, n]
            Symmetric matrix.
        """
        eigvals, eigvecs = gs.linalg.eigh(x)
        if check_positive:
            if gs.any(gs.cast(eigvals, gs.float32) < 0.):
                logging.warning(
                    'Negative eigenvalue encountered in'
                    ' {}'.format(function.__name__))
        eigvals = function(eigvals)
        eigvals = algebra_utils.from_vector_to_diagonal_matrix(eigvals)
        transp_eigvecs = Matrices.transpose(eigvecs)
        reconstuction = gs.matmul(eigvecs, eigvals)
        reconstuction = gs.matmul(reconstuction, transp_eigvecs)
        return reconstuction
