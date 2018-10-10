"""
The space of matrices (m, n), which is the Euclidean space R^{mn}.
"""

import geomstats.backend as gs

from geomstats.euclidean_space import EuclideanSpace


class MatrixSpace(EuclideanSpace):
    """Class for the space of matrices (m, n)."""

    def __init__(self, m, n):
        assert isinstance(m, int) and isinstance(n, int) and m > 0 and n > 0
        super().init(dimension=m*n)
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
