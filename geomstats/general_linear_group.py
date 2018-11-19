"""
The General Linear Group, i.e. the matrix group GL(n).
"""

import scipy.linalg

import geomstats.backend as gs

from geomstats.lie_group import LieGroup
from geomstats.matrices_space import MatricesSpace


class GeneralLinearGroup(LieGroup, MatricesSpace):
    """
    Class for the General Linear Group, i.e. the matrix group GL(n).


    Note: The default representation for elements of GL(n)
    are matrices.
    For now, SO(n) and SE(n) elements are represented
    by a vector by default.
    """

    def __init__(self, n):
        assert isinstance(n, int) and n > 0
        LieGroup.__init__(self, dimension=n*n)
        MatricesSpace.__init__(self, m=n, n=n)

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
        mat_rank = gs.linalg.matrix_rank(mat)
        mat_rank = gs.to_ndarray(mat_rank, to_ndim=1)

        return gs.equal(mat_rank, self.n)

    def compose(self, mat_a, mat_b):
        """
        Matrix composition.
        """
        mat_a = gs.to_ndarray(mat_a, to_ndim=3)
        mat_b = gs.to_ndarray(mat_b, to_ndim=3)
        composition = gs.einsum('nij,njk->nik', mat_a, mat_b)
        return composition

    def inverse(self, mat):
        """
        Matrix inverse.
        """
        mat = gs.to_ndarray(mat, to_ndim=3)
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
