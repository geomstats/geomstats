"""Base class for matrix Lie groups."""

import numpy as np
import scipy.linalg

from geomstats.lie_group import LieGroup


class MatrixLieGroup(LieGroup):
    """
    Base class for matrix Lie groups,
    where each element is represented by a matrix by default.

    Note: for now, SO(n) and SE(n) elements are represented
    by a vector by default.
    """

    def __init__(self, dimension, n):
        super(MatrixLieGroup, self).__init__(
                                      dimension=dimension,
                                      identity=np.eye(n))
        self.n = n

    def belongs(self, mat):
        """
        Check if mat belongs to the Matrix Lie group.

        Note:
        - By default, check that the matrix is invertible.
        - Need override for any matrix Lie group
        that is not GL(n).
        """
        mat_rank = np.linalg.matrix_rank(mat)

        return mat_rank == self.n

    def compose(self, mat_a, mat_b):
        """
        Matrix composition.
        """
        return np.dot(mat_a, mat_b)

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
