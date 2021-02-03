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
    """Class for the vector space of symmetric matrices of size n.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices: n x n.
    """

    def __init__(self, n, **kwargs):
        super(SymmetricMatrices, self).__init__(
            dim=int(n * (n + 1) / 2),
            embedding_manifold=Matrices(n, n))
        self.n = n

    def belongs(self, mat, atol=TOLERANCE):
        """Check if mat belongs to the vector space of symmetric matrices.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix to check.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if mat is a symmetric matrix.
        """
        check_shape = self.embedding_manifold.belongs(mat)
        return gs.logical_and(check_shape, Matrices.is_symmetric(mat, atol))

    def get_basis(self):
        """Compute the basis of the vector space of symmetric matrices."""
        basis = []
        for row in gs.arange(self.n):
            for col in gs.arange(row, self.n):
                if row == col:
                    indices = [(row, row)]
                    values = [1.]
                else:
                    indices = [(row, col), (col, row)]
                    values = [1., 1.]
                basis.append(gs.array_from_sparse(
                    indices, values, (self.n, ) * 2))
        basis = gs.stack(basis)
        return basis

    basis = property(get_basis)

    @staticmethod
    def projection(point):
        """Make a matrix symmetric, by averaging with its transpose.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        sym : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        return Matrices.to_symmetric(point)

    @staticmethod
    def to_vector(mat):
        """Convert a symmetric matrix into a vector.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        vec : array-like, shape=[..., n(n+1)/2]
            Vector.
        """
        if not gs.all(Matrices.is_symmetric(mat)):
            logging.warning('non-symmetric matrix encountered.')
        mat = Matrices.to_symmetric(mat)
        return gs.triu_to_vec(mat)

    @staticmethod
    @geomstats.vectorization.decorator(['vector', 'else'])
    def from_vector(vec, dtype=gs.float32):
        """Convert a vector into a symmetric matrix.

        Parameters
        ----------
        vec : array-like, shape=[..., n(n+1)/2]
            Vector.

        Returns
        -------
        mat : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
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
    def expm(cls, mat):
        """
        Compute the matrix exponential for a symmetric matrix.

        Parameters
        ----------
        mat : array_like, shape=[..., n, n]
            Symmetric matrix.

        Returns
        -------
        exponential : array_like, shape=[..., n, n]
            Exponential of mat.
        """
        return cls.apply_func_to_eigvals(mat, gs.exp)

    @classmethod
    def powerm(cls, mat, power):
        """
        Compute the matrix power.

        Parameters
        ----------
        mat : array_like, shape=[..., n, n]
            Symmetric matrix with non-negative eigenvalues.
        power : float
            Power at which mat will be raised.

        Returns
        -------
        powerm : array_like, shape=[..., n, n]
            Matrix power of mat.
        """
        def _power(eigvals):
            return gs.power(eigvals, power)
        return cls.apply_func_to_eigvals(mat, _power, check_positive=True)

    @staticmethod
    def apply_func_to_eigvals(mat, function, check_positive=False):
        """
        Apply function to eigenvalues and reconstruct the matrix.

        Parameters
        ----------
        mat : array_like, shape=[..., n, n]
            Symmetric matrix.
        function : callable
            Function to apply to eigenvalues.

        Returns
        -------
        mat : array_like, shape=[..., n, n]
            Symmetric matrix.
        """
        eigvals, eigvecs = gs.linalg.eigh(mat)
        if check_positive:
            if gs.any(gs.cast(eigvals, gs.float32) < 0.):
                logging.warning(
                    'Negative eigenvalue encountered in'
                    ' {}'.format(function.__name__))
        eigvals = function(eigvals)
        eigvals = algebra_utils.from_vector_to_diagonal_matrix(eigvals)
        transp_eigvecs = Matrices.transpose(eigvecs)
        reconstruction = gs.matmul(eigvecs, eigvals)
        reconstruction = gs.matmul(reconstruction, transp_eigvecs)
        return reconstruction
