"""Module providing the SkewSymmetricMatrices class.

This is the Lie algebra of the Special Orthogonal Group.
As basis we choose the matrices with a single 1 on the upper triangular part
of the matrices (and a -1 in its lower triangular part).
"""
import geomstats.backend as gs
from geomstats.geometry.lie_algebra import MatrixLieAlgebra


class SkewSymmetricMatrices(MatrixLieAlgebra):
    """Class for skew symmetric matrices."""

    def __init__(self, n):
        """Instantiate the class.

        Parameters
        ----------
        n: int
            the amount of columns / rows
        """
        dimension = int(n * (n - 1) / 2)
        super(SkewSymmetricMatrices, self).__init__(dimension, n)

        self.basis = gs.zeros((dimension, n, n))
        loop_index = 0

        for row in gs.arange(n - 1):
            for col in gs.arange(row + 1, n):
                self.basis[loop_index, row, col] = 1
                self.basis[loop_index, col, row] = -1
                loop_index += 1

    def basis_representation(self, matrix_representation):
        """Calculate the coefficients of given matrix in the basis.

        Parameters
        ----------
        matrix_representation: array-like, shape=[n_sample, n, n]

        Returns
        -------
        basis_representation: array-like, shape=[n_sample, dimension]
        """
        old_shape = gs.shape(matrix_representation)
        as_vector = gs.reshape(matrix_representation, (old_shape[0], -1))
        upper_tri_indices = gs.reshape(
            gs.arange(0, self.n ** 2), (self.n, self.n)
        )[gs.triu_indices(self.n, k=1)]

        return as_vector[:, upper_tri_indices]
