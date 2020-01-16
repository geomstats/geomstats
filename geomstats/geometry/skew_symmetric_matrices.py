import geomstats.backend as gs
from geomstats.geometry.lie_algebra import MatrixLieAlgebra


class SkewSymmetricMatrices(MatrixLieAlgebra):

    def __init__(self, dimension, n):
        if 2 * dimension != n * (n - 1):
            raise ValueError("""Dimension and Matrix space do not fit together
                                for SkewSymmetricMatrices""")

        super(SkewSymmetricMatrices, self).__init__(dimension, n)

        self.basis = gs.zeros((dimension, n, n))
        loop_index = 0

        for i in gs.arange(n - 1):  # rows
            for j in gs.arange(i + 1, n):  # columns
                self.basis[loop_index, i, j] = 1
                self.basis[loop_index, j, i] = -1
                loop_index += 1

    def basis_representation(self, matrix_representation):
        """
        Parameters
        ----------
        matrix_representation: array-like, shape=[n_sample, n, n]

        Returns
        ------
        basis_representation: array-like, shape=[n_sample, dimension]
        """
        x = gs.reshape(matrix_representation, (-1, self.n ** 2))
        return x[:, [1, 2, 5]]
