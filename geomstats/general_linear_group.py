"""
The General Linear Group, i.e. the matrix group GL(n).
"""

import scipy.linalg

import geomstats.backend as gs

from geomstats.lie_group import LieGroup


class GeneralLinearGroup(LieGroup):
    """
    Class for the General Linear Group, i.e. the matrix group GL(n).


    Note: The default representation for elements of GL(n)
    are matrices.
    For now, SO(n) and SE(n) elements are represented
    by a vector by default.
    """

    def __init__(self, n):
        assert isinstance(n, int) and n > 0
        super(GeneralLinearGroup, self).__init__(
                                      dimension=n*n)
        self.n = n

    @property
    def identity(self):
        return gs.eye(self.n)

    def belongs(self, mat):
        """
        Check if mat belongs to GL(n).
        """
        mat = gs.to_ndarray(mat, to_ndim=3)
        n_mats, _, _ = mat.shape

        mat_rank = gs.zeros((n_mats, 1))
        for i in range(n_mats):
            mat_rank[i] = gs.linalg.matrix_rank(mat[i])

        return mat_rank == self.n

    def compose(self, mat_a, mat_b):
        """
        Matrix composition.
        """
        return gs.matmul(mat_a, mat_b)

    def inverse(self, mat):
        """
        Matrix inverse.
        """
        return gs.linalg.inv(mat)

    def group_exp_from_identity(self, tangent_vec):
        """
        Compute the group exponential
        of tangent vector tangent_vec from the identity.
        """
        if tangent_vec.ndim == 2:
            return scipy.linalg.expm(tangent_vec)

        exp = gs.zeros_like(tangent_vec)
        n_vecs, _, _ = tangent_vec.shape
        for i in range(n_vecs):
            exp[i] = scipy.linalg.expm(tangent_vec[i])

        return exp

    def group_log_from_identity(self, point):
        """
        Compute the group logarithm
        of the point point from the identity.
        """
        if point.ndim == 2:
            return scipy.linalg.logm(point)

        log = gs.zeros_like(point)
        n_points, _, _ = point.shape
        for i in range(n_points):
            log[i] = scipy.linalg.logm(point[i])

        return log
