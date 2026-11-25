"""The vector space of Hermitian matrices.

Lead author: Yann Cabanes.
"""

import logging

import geomstats.backend as gs
from geomstats.geometry.base import MatrixVectorSpace
from geomstats.geometry.complex_matrices import ComplexMatrices, ComplexMatricesMetric
from geomstats.geometry.manifold import _get_default_dtype
from geomstats.geometry.matrices import Matrices

logger = logging.getLogger(__name__)


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

        logger.warning("Negative eigenvalue encountered in %s", name)

    return_list = True
    if not isinstance(function, list):
        function = [function]
        return_list = False
    reconstruction = []

    if gs.is_complex(mat):
        transp_eigvecs = ComplexMatrices.transconjugate(eigvecs)
    else:
        transp_eigvecs = gs.matrix_transpose(eigvecs)

    for fun in function:
        reconstruction.append(
            gs.matmul(eigvecs * fun(eigvals)[..., None, :], transp_eigvecs)
        )
    return reconstruction if return_list else reconstruction[0]


class HermitianMatrices(MatrixVectorSpace):
    """Class for the vector space of Hermitian matrices of size n.

    It is a real vector space, whose elements are complex matrices.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices: n x n.
    equip : bool
        Whether to equip space with default metric.
    """

    def __init__(self, n, equip=True):
        self.n = n
        super().__init__(
            dim=n**2,
            shape=(n, n),
            equip=equip,
            field="real",
            complex_repr=True,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return ComplexMatricesMetric

    def _create_basis(self):
        """Compute the basis of the vector space of Hermitian matrices.

        The basis is made of the diagonal matrices :math:`E_{kk}`, then of
        :math:`E_{kl} + E_{lk}` and finally of :math:`i(E_{kl} - E_{lk})`,
        for :math:`k < l`.
        """
        n = self.n
        n_upper = n * (n - 1) // 2
        real_indices = [(k, k, k) for k in range(n)]
        real_values = [1.0] * n
        imag_indices, imag_values = [], []
        index = n
        for row in range(n):
            for col in range(row + 1, n):
                real_indices.extend([(index, row, col), (index, col, row)])
                real_values.extend([1.0, 1.0])
                imag_indices.extend(
                    [(index + n_upper, row, col), (index + n_upper, col, row)]
                )
                imag_values.extend([1.0, -1.0])
                index += 1

        shape = (self.dim, n, n)
        real_part = gs.array_from_sparse(real_indices, real_values, shape)
        imag_part = (
            gs.array_from_sparse(imag_indices, imag_values, shape)
            if imag_indices
            else gs.zeros(shape)
        )
        cdtype = _get_default_dtype(self.complex_repr)
        return gs.cast(real_part, cdtype) + 1j * gs.cast(imag_part, cdtype)

    def basis_representation(self, matrix_representation):
        """Compute the coefficients of Hermitian matrices in the basis.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Hermitian matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., n**2]
            Real coefficients in the basis.
        """
        rows, cols = gs.triu_indices(self.n, k=1)
        upper = matrix_representation[..., rows, cols]
        return gs.concatenate(
            [
                gs.real(Matrices.diagonal(matrix_representation)),
                gs.real(upper),
                gs.imag(upper),
            ],
            axis=-1,
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
            Hermitian matrix.
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
