"""The vector space of symmetric matrices.

Lead author: Yann Thanwerdas.
"""

import logging

import geomstats.backend as gs
from geomstats import algebra_utils
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.matrices import Matrices, MatricesMetric


class SymmetricMatrices(VectorSpace):
    """Class for the vector space of symmetric matrices of size n.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices: n x n.
    """

    def __init__(self, n, equip=True):
        super().__init__(dim=int(n * (n + 1) / 2), shape=(n, n), equip=equip)
        self.n = n

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return MatricesMetric

    def _create_basis(self):
        """Compute the basis of the vector space of symmetric matrices."""
        indices, values = [], []
        k = -1
        for row in range(self.n):
            for col in range(row, self.n):
                k += 1
                if row == col:
                    indices.append((k, row, row))
                    values.append(1.0)
                else:
                    indices.extend([(k, row, col), (k, col, row)])
                    values.extend([1.0, 1.0])

        return gs.array_from_sparse(indices, values, (k + 1, self.n, self.n))

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a matrix is symmetric.

        Parameters
        ----------
        point : array-like, shape=[.., n, n]
            Point to test.
        atol : float
            Tolerance to evaluate equality with the transpose.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        belongs = super().belongs(point)
        if gs.any(belongs):
            is_symmetric = Matrices.is_symmetric(point, atol)
            return gs.logical_and(belongs, is_symmetric)
        return belongs

    def projection(self, point):
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

    def random_point(self, n_samples=1, bound=1.0):
        """Sample a symmetric matrix.

        Samples from a uniform distribution in a box and then converts to symmetric.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0

        Returns
        -------
        point : array-like, shape=[..., n, n]
           Sample.
        """
        sample = super().random_point(n_samples, bound)
        return Matrices.to_symmetric(sample)

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
        return gs.triu_to_vec(mat)

    @staticmethod
    def from_vector(vec):
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
        mat_dim = (gs.sqrt(8.0 * vec_dim + 1) - 1) / 2
        if mat_dim != int(mat_dim):
            raise ValueError(
                "Invalid input dimension, it must be of the form"
                "(n_samples, n * (n + 1) / 2)"
            )
        mat_dim = int(mat_dim)
        shape = (mat_dim, mat_dim)
        mask = 2 * gs.ones(shape) - gs.eye(mat_dim)
        indices = list(zip(*gs.triu_indices(mat_dim)))
        if gs.ndim(vec) == 1:
            upper_triangular = gs.array_from_sparse(indices, vec, shape)
        else:
            upper_triangular = gs.stack(
                [gs.array_from_sparse(indices, data, shape) for data in vec]
            )

        mat = Matrices.to_symmetric(upper_triangular) * mask
        return mat

    @classmethod
    def expm(cls, mat):
        """Compute the matrix exponential for a symmetric matrix.

        Parameters
        ----------
        mat : array_like, shape=[..., n, n]
            Symmetric matrix.

        Returns
        -------
        exponential : array_like, shape=[..., n, n]
            Exponential of mat.
        """
        n = mat.shape[-1]
        dim_3_mat = gs.reshape(mat, [-1, n, n])
        expm = cls.apply_func_to_eigvals(dim_3_mat, gs.exp)
        expm = gs.reshape(expm, mat.shape)
        return expm

    @classmethod
    def powerm(cls, mat, power):
        """
        Compute the matrix power.

        Parameters
        ----------
        mat : array_like, shape=[..., n, n]
            Symmetric matrix with non-negative eigenvalues.
        power : float, list
            Power at which mat will be raised. If a list of powers is passed,
            a list of results will be returned.

        Returns
        -------
        powerm : array_like or list of arrays, shape=[..., n, n]
            Matrix power of mat.
        """
        if isinstance(power, list):
            power_ = [lambda ev, p=p: gs.power(ev, p) for p in power]
        else:

            def power_(ev):
                return gs.power(ev, power)

        return cls.apply_func_to_eigvals(mat, power_, check_positive=False)

    @staticmethod
    def apply_func_to_eigvals(mat, function, check_positive=False):
        """
        Apply function to eigenvalues and reconstruct the matrix.

        Parameters
        ----------
        mat : array_like, shape=[..., n, n]
            Symmetric matrix.
        function : callable, list of callables
            Function to apply to eigenvalues. If a list of functions is passed,
            a list of results will be returned.
        check_positive : bool
            Whether to check positivity of the eigenvalues.
            Optional. Default: False.

        Returns
        -------
        mat : array_like, shape=[..., n, n]
            Symmetric matrix.
        """
        eigvals, eigvecs = gs.linalg.eigh(mat)
        if check_positive and gs.any(eigvals < 0.0):
            try:
                name = function.__name__
            except AttributeError:
                name = function[0].__name__

            logging.warning("Negative eigenvalue encountered in %s", name)

        return_list = True
        if not isinstance(function, list):
            function = [function]
            return_list = False
        reconstruction = []
        transp_eigvecs = Matrices.transpose(eigvecs)
        for fun in function:
            eigvals_f = fun(eigvals)
            eigvals_f = algebra_utils.from_vector_to_diagonal_matrix(eigvals_f)
            reconstruction.append(Matrices.mul(eigvecs, eigvals_f, transp_eigvecs))
        return reconstruction if return_list else reconstruction[0]
