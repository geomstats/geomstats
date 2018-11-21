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

    def group_exp_from_identity(self, tangent_vec, point_type=None):
        """
        Group exponential of the Lie group of
        all invertible matrices has a straight-forward
        computation for symmetric positive definite matrices.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        n_tangent_vecs, mat_dim, _ = tangent_vec.shape

        if gs.all(self.is_symmetric(tangent_vec)):
            tangent_vec = self.make_symmetric(tangent_vec)
            [eigenvalues, vectors] = gs.linalg.eigh(tangent_vec)
            exp_eigenvalues = gs.exp(eigenvalues)

            aux = gs.einsum('ijk,ik->ijk', vectors, exp_eigenvalues)
            group_exp = gs.einsum('ijk,ilk->ijl', aux, vectors)

            group_exp = gs.to_ndarray(group_exp, to_ndim=3)

        else:
            group_exp = gs.expm(tangent_vec)

        return group_exp.real

    def group_log_from_identity(self, point, point_type=None):
        """
        Group logarithm of the Lie group of
        all invertible matrices has a straight-forward
        computation for symmetric positive definite matrices.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        if gs.all(self.is_symmetric(point)):
            point = self.make_symmetric(point)
            [eigenvalues, vectors] = gs.linalg.eigh(point)

            log_eigenvalues = gs.log(eigenvalues.real)

            aux = gs.einsum('ijk,ik->ijk', vectors, log_eigenvalues)
            group_log = gs.einsum('ijk,ilk->ijl', aux, vectors)

        else:
            group_log = gs.logm(point)

        return group_log.real
