"""The vector space of Hermitian matrices.

Lead author: Yann Cabanes.
"""
import logging

import geomstats.backend as gs
from geomstats import algebra_utils as utils
from geomstats.geometry.base import ComplexMatrixVectorSpace
from geomstats.geometry.complex_matrices import ComplexMatrices, ComplexMatricesMetric
from geomstats.geometry.matrices import Matrices


def expmh(mat):
    """Compute the matrix exponential for a Hermitian matrix.

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
    expm = apply_func_to_eigvalsh(dim_3_mat, gs.exp)
    return gs.reshape(expm, mat.shape)


def powermh(mat, power):
    """Compute the matrix power for a Hermitian matrix.

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

    return apply_func_to_eigvalsh(mat, power_, check_positive=False)


def apply_func_to_eigvalsh(mat, function, check_positive=False):
    """Apply function to eigenvalues and reconstruct the matrix.

    Parameters
    ----------
    mat : array_like, shape=[..., n, n]
        Hermitian matrix.
    function : callable, list of callables
        Function to apply to eigenvalues. If a list of functions is passed,
        a list of results will be returned.
    check_positive : bool
        Whether to check positivity of the eigenvalues.
        Optional. Default: False.

    Returns
    -------
    mat : array_like, shape=[..., n, n]
        Hermitian matrix.
    """
    eigvals, eigvecs = gs.linalg.eigh(mat)
    if check_positive and gs.any(gs.cast(eigvals, gs.get_default_dtype()) < 0.0):
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

    if gs.is_complex(mat):
        transp_eigvecs = ComplexMatrices.transconjugate(eigvecs)
    else:
        transp_eigvecs = Matrices.transpose(eigvecs)

    for fun in function:
        eigvals_f = fun(eigvals)
        eigvals_f = utils.from_vector_to_diagonal_matrix(eigvals_f)
        reconstruction.append(Matrices.mul(eigvecs, eigvals_f, transp_eigvecs))
    return reconstruction if return_list else reconstruction[0]


class HermitianMatrices(ComplexMatrixVectorSpace):
    """Class for the vector space of Hermitian matrices of size n.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices: n x n.
    """

    def __init__(self, n, equip=True):
        super().__init__(dim=n * (n + 1) - n, shape=(n, n), equip=equip)
        self.n = n

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return ComplexMatricesMetric

    def _create_basis(self):
        """Compute the basis of the vector space of symmetric matrices.

        Returns
        -------
        basis : array-like, shape=[dim, n, n]
        """
        diagonal = []
        real_part = []
        complex_part = []
        for row in gs.arange(self.n):
            for col in gs.arange(row, self.n):
                if row == col:
                    indices = [(row, row)]
                    values = [1.0 + 0j]
                    diagonal.append(
                        gs.array_from_sparse(indices, values, (self.n,) * 2)
                    )
                else:
                    indices = [(row, col), (col, row)]
                    values = [1.0 + 0j, 1.0 + 0j]
                    real_part.append(
                        gs.array_from_sparse(indices, values, (self.n,) * 2)
                    )
                    values = [1j, -1j]
                    complex_part.append(
                        gs.array_from_sparse(indices, values, (self.n,) * 2)
                    )
        return gs.vstack(
            [gs.stack(diagonal), gs.stack(real_part), gs.stack(complex_part)]
        )

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a matrix is Hermitian.

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
            is_hermitian = ComplexMatrices.is_hermitian(point, atol)
            return gs.logical_and(belongs, is_hermitian)
        return belongs

    @staticmethod
    def projection(point):
        """Make a matrix Hermitian, by averaging with its transconjugate.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        herm : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        return ComplexMatrices.to_hermitian(point)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample a Hermitian matrix.

        Points are generated by sampling complex matrices from a uniform distribution
        in a box and averaging with the transconjugate.

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
        cdtype = gs.get_default_cdtype()
        size = self.shape
        if n_samples != 1:
            size = (n_samples,) + self.shape
        point = gs.cast(
            bound * (gs.random.rand(*size) - 0.5) * 2**0.5,
            dtype=cdtype,
        ) + 1j * gs.cast(
            bound * (gs.random.rand(*size) - 0.5) * 2**0.5,
            dtype=cdtype,
        )
        return ComplexMatrices.to_hermitian(point)

    @staticmethod
    def basis_representation(matrix_representation):
        """Convert a Hermitian matrix into a vector.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., n(n+1)/2]
            Vector.
        """
        diag = Matrices.diagonal(matrix_representation)

        up_triang = gs.triu_to_vec(matrix_representation, k=1)
        real_part = gs.real(up_triang)
        complex_part = gs.imag(up_triang)

        vec = gs.hstack([diag, real_part, complex_part])

        return vec
