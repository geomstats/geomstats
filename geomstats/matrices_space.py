"""
The space of matrices (m, n), which is the Euclidean space R^{mn}.
"""

import geomstats.backend as gs

from geomstats.euclidean_space import EuclideanSpace


TOLERANCE = 1e-5


class MatricesSpace(EuclideanSpace):
    """Class for the space of matrices (m, n)."""

    def __init__(self, m, n):
        assert isinstance(m, int) and isinstance(n, int) and m > 0 and n > 0
        super(MatricesSpace, self).__init__(dimension=m*n)
        self.m = m
        self.n = n
        self.default_point_type = 'matrix'

    def belongs(self, point):
        """
        Check if point belongs to the Matrix space.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        _, mat_dim_1, mat_dim_2 = point.shape
        return mat_dim_1 == self.m & mat_dim_2 == self.n

    @staticmethod
    def vector_from_matrix(matrix):
        """
        Conversion function from (_, m, n) to (_, mn).
        """
        matrix = gs.to_ndarray(matrix, to_ndim=3)
        n_mats, m, n = matrix.shape
        return gs.reshape(matrix, (n_mats, m*n))

    @staticmethod
    def is_symmetric(matrix, tolerance=TOLERANCE):
        """Check if a matrix is symmetric."""
        matrix = gs.to_ndarray(matrix, to_ndim=3)
        n_mats, m, n = matrix.shape
        assert m == n
        matrix_transpose = gs.transpose(matrix, axes=(0, 2, 1))

        mask = gs.isclose(matrix, matrix_transpose, atol=tolerance)
        mask = gs.all(mask, axis=(1, 2))

        return mask

    @staticmethod
    def make_symmetric(matrix):
        """Make a matrix fully symmetric to avoid numerical issues."""
        matrix = gs.to_ndarray(matrix, to_ndim=3)
        n_mats, m, n = matrix.shape
        assert m == n
        matrix = gs.to_ndarray(matrix, to_ndim=3)
        return (matrix + gs.transpose(matrix, axes=(0, 2, 1))) / 2

    def random_uniform(self, n_samples=1):
        point = gs.random.rand(n_samples, self.m, self.n)
        return point
