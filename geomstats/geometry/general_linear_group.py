"""
The General Linear Group, i.e. the matrix group GL(n).
"""

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

    def get_identity(self, point_type=None):
        if point_type is None:
            point_type = self.default_point_type
        if point_type == 'matrix':
            return gs.eye(self.n)
        else:
            raise NotImplementedError(
                'The identity of the general linear group is not'
                ' implemented for a point_type that is not \'matrix\'.')
    identity = property(get_identity)

    def belongs(self, mat):
        """
        Check if mat belongs to GL(n).
        """
        mat = gs.to_ndarray(mat, to_ndim=3)

        det = gs.linalg.det(mat)
        belongs = ~gs.isclose(det, 0.)

        belongs = gs.to_ndarray(belongs, to_ndim=1)
        belongs = gs.to_ndarray(belongs, to_ndim=2, axis=1)

        return belongs

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

    def group_exp_from_identity(self, tangent_vec, point_type=None):
        """
        Group exponential of the Lie group of
        all invertible matrices at the identity.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        group_exp = gs.linalg.expm(tangent_vec)

        return gs.real(group_exp)

    def group_exp_not_from_identity(
            self, tangent_vec, base_point, point_type=None):
        """
        Group exponential of the Lie group of
        all invertible matrices.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        base_point = gs.to_ndarray(base_point, to_ndim=3)

        tangent_vec_at_identity = self.compose(
            self.inverse(base_point), tangent_vec)

        group_exp_from_identity = self.group_exp_from_identity(
                tangent_vec_at_identity)

        group_exp = self.compose(
            base_point, group_exp_from_identity)

        return group_exp

    def group_log_from_identity(self, point, point_type=None):
        """
        Group logarithm of the Lie group of
        all invertible matrices at the identity.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        group_log = gs.linalg.logm(point)

        return gs.real(group_log)

    def group_log_not_from_identity(
            self, point, base_point, point_type=None):
        """
        Group logarithm of the Lie group of
        all invertible matrices.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        base_point = gs.to_ndarray(base_point, to_ndim=3)

        point_near_identity = self.compose(
            self.inverse(base_point), point)

        group_log_from_identity = self.group_log_from_identity(
            point_near_identity)

        group_log = self.compose(
            base_point, group_log_from_identity)

        return group_log
