import geomstats.backend as gs
from geomstats.geometry.lie_algebra import MatrixLieAlgebra


class SkewSymmetricMatrices(MatrixLieAlgebra):
    def __init__(self, n):
        dimension = int(n * (n - 1) / 2)
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
        old_shape = gs.shape(matrix_representation)
        as_vector = gs.reshape(matrix_representation, (old_shape[0], -1))
        upper_tri_indices = gs.reshape(
            gs.arange(0, self.n ** 2), (self.n, self.n)
        )[gs.triu_indices(self.n, k=1)]

        return as_vector[:, upper_tri_indices]
