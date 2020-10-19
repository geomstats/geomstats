"""Module providing the SkewSymmetricMatrices class.

This is the Lie algebra of the Special Orthogonal Group.
As basis we choose the matrices with a single 1 on the upper triangular part
of the matrices (and a -1 in its lower triangular part).
"""
import geomstats.backend as gs
from geomstats.geometry.lie_algebra import MatrixLieAlgebra
from geomstats.geometry.matrices import Matrices


TOLERANCE = 1e-8


class SkewSymmetricMatrices(MatrixLieAlgebra):
    """Class for skew-symmetric matrices.

    Parameters
    ----------
    n : int
        Number of rows and columns.
    """

    def __init__(self, n):
        dimension = int(n * (n - 1) / 2)
        super(SkewSymmetricMatrices, self).__init__(dimension, n)

        self.basis = gs.zeros((dimension, n, n))

        basis = []
        for row in gs.arange(n - 1):
            for col in gs.arange(row + 1, n):
                basis.append(gs.array_from_sparse(
                    [(row, col), (col, row)], [1., -1.], (n, n)))
        self.basis = gs.stack(basis)

    def belongs(self, mat, atol=TOLERANCE):
        """Evaluate if mat is a skew-symmetric matrix.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Square matrix to check.
        atol : float
            Tolerance for the equality evaluation.
            Optional, default: TOLERANCE.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if matrix is skew symmetric.
        """
        return Matrices(self.n, self.n).is_skew_symmetric(mat=mat, atol=atol)

    def basis_representation(self, matrix_representation):
        """Calculate the coefficients of given matrix in the basis.

        Compute a 1d-array that corresponds to the input matrix in the basis
        representation.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., dim]
            Representation in the basis.
        """
        old_shape = gs.shape(matrix_representation)
        as_vector = gs.reshape(matrix_representation, (old_shape[0], -1))
        upper_tri_indices = gs.reshape(
            gs.arange(0, self.n ** 2), (self.n, self.n)
        )[gs.triu_indices(self.n, k=1)]

        return as_vector[:, upper_tri_indices]
