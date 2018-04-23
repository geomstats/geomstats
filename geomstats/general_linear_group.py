"""
Base class for the General Lie Group,
i.e. the matrix group GL(n).
"""

import numpy as np
import scipy.linalg

from geomstats.lie_group import LieGroup
import geomstats.vectorization as vectorization


class GeneralLinearGroup(LieGroup):
    """
    Base class for the General Lie Group,
    i.e. the matrix group GL(n).

    Note: for now, SO(n) and SE(n) elements are represented
    by a vector by default.
    """

    def __init__(self, n):
        super(GeneralLinearGroup, self).__init__(
                                      dimension=n*n,
                                      identity=np.eye(n))
        self.n = n

    def belongs(self, mat):
        """
        Check if mat belongs to GL(n).
        """
        mat = vectorization.to_ndarray(mat, to_ndim=3)
        n_mats, _, _ = mat.shape

        mat_rank = np.zeros((n_mats, 1))
        for i in range(n_mats):
            mat_rank[i] = np.linalg.matrix_rank(mat[i])

        return mat_rank == self.n

    def compose(self, mat_a, mat_b):
        """
        Matrix composition.
        """
        return np.matmul(mat_a, mat_b)

    def inverse(self, mat):
        """
        Matrix inverse.
        """
        return np.linalg.inv(mat)

    def group_exp_from_identity(self, tangent_vec):
        """
        Compute the group exponential
        of tangent vector tangent_vec from the identity.
        """
        return scipy.linalg.expm(tangent_vec)

    def group_log_from_identity(self, point):
        """
        Compute the group logarithm
        of the point point from the identity.
        """
        return scipy.linalg.logm(point)
