import geomstats.backend as gs
from geomstats.geometry.lie_algebra import MatrixLieAlgebra


class SkewSymmetricMatrices(MatrixLieAlgebra):

    def __init__(self, dimension, n):
        if 2 * dimension != n * (n - 1):
            raise ValueError("""Dimension and Matrix space do not fit together
                                for SkewSymmetricMatrices""")

        super(SkewSymmetricMatrices, self).__init__(dimension, n)

        self.basis = gs.zeros((3, n, n))

        if n == 3:
            self.basis = gs.array([
                [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
                [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, -1, 0]]])
        else:
            raise NotImplementedError("""SkewSymmetricMatrices is only
                                         implemented for n = 3""")

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
