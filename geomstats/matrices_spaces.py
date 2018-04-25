"""
The space of matrices (m, n), which is the Euclidean space
R^{mn}.
"""

import numpy as np

from geomstats.euclidean_space import EuclideanSpace
import geomstats.vectorization as vectorization


class MatrixSpace(EuclideanSpace):
    """The space of matrices (m, n)."""

    def __init__(self, m, n):
        assert m > 0 & n > 0
        super().init(dimension=m*n)
        self.m = m
        self.n = n

    def belongs(self, point):
        """
        Check if point belongs to the Matrix space.
        """
        point = vectorization.to_ndarray(point, to_ndim=3)
        _, mat_dim_1, mat_dim_2 = point.shape
        return mat_dim_1 == self.m & mat_dim_2 == self.n

    def vector_from_matrix(self, matrix):
        """
        Conversion function from (_, m, n) to (_, mn).
        """
        matrix = vectorization.to_ndarray(matrix, to_ndim=3)
        n_mats, m, n = matrix.shape
        return np.reshape(matrix, (n_mats, m*n))
