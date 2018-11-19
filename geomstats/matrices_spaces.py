"""
The space of matrices (m, n), which is the Euclidean space R^{mn}.
"""

import geomstats.backend as gs

from geomstats.euclidean_space import EuclideanSpace


TOLERANCE = 1e-5


class MatrixSpace(EuclideanSpace):
    """Class for the space of matrices (m, n)."""

    def __init__(self, m, n):
        assert isinstance(m, int) and isinstance(n, int) and m > 0 and n > 0
        super(MatrixSpace, self).__init__(dimension=m*n)
        self.m = m
        self.n = n

    def belongs(self, point):
        """
        Check if point belongs to the Matrix space.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        _, mat_dim_1, mat_dim_2 = point.shape
        return mat_dim_1 == self.m & mat_dim_2 == self.n

    def vector_from_matrix(self, matrix):
        """
        Conversion function from (_, m, n) to (_, mn).
        """
        matrix = gs.to_ndarray(matrix, to_ndim=3)
        n_mats, m, n = matrix.shape
        return gs.reshape(matrix, (n_mats, m*n))

    def is_symmetric(self, matrix, tolerance=TOLERANCE):
        """Check if a matrix is symmetric."""
        assert self.m == self.n

        matrix = gs.to_ndarray(matrix, to_ndim=3)
        matrix_transpose = gs.transpose(matrix, axes=(0, 2, 1))

        mask = gs.isclose(matrix, matrix_transpose, atol=tolerance)
        mask = gs.all(mask, axis=(1, 2))

        return mask

    def make_symmetric(self, matrix):
        """Make a matrix fully symmetric to avoid numerical issues."""
        assert self.m == self.n
        matrix = gs.to_ndarray(matrix, to_ndim=3)
        return (matrix + gs.transpose(matrix, axes=(0, 2, 1))) / 2

    def sqrtm(self, matrix):
        assert self.m == self.n
        matrix = gs.to_ndarray(matrix, to_ndim=3)

        if gs.all(self.is_symmetric(matrix)):
            [eigenvalues, vectors] = gs.linalg.eigh(matrix)
        else:
            [eigenvalues, vectors] = gs.linalg.eig(matrix)

        sqrt_eigenvalues = gs.sqrt(eigenvalues)

        aux = gs.einsum('ijk,ik->ijk', vectors, sqrt_eigenvalues)
        sqrt_mat = gs.einsum('ijk,ilk->ijl', aux, vectors)

        sqrt_mat = gs.to_ndarray(sqrt_mat, to_ndim=3)
        return sqrt_mat.real
