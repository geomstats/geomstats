"""
Unit tests for General Linear group.
"""

import unittest

import geomstats.backend as gs

from geomstats.general_linear_group import GeneralLinearGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

RTOL = 1e-5


class TestGeneralLinearGroupMethods(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        n = 3
        self.group = GeneralLinearGroup(n=n)
        # We generate invertible matrices using so3_group
        self.so3_group = SpecialOrthogonalGroup(n=n)

    def test_belongs(self):
        """
        A rotation matrix belongs to the matrix Lie group
        of invertible matrices.
        """
        rot_vec = self.so3_group.random_uniform()
        rot_mat = self.so3_group.matrix_from_rotation_vector(rot_vec)

        self.assertTrue(self.group.belongs(rot_mat))

    def test_compose(self):
        # 1. Composition by identity, on the right
        # Expect the original transformation
        rot_vec_1 = self.so3_group.random_uniform()
        mat_1 = self.so3_group.matrix_from_rotation_vector(rot_vec_1)

        result_1 = self.group.compose(mat_1, self.group.identity)
        expected_1 = mat_1

        self.assertTrue(gs.allclose(result_1, expected_1))

        # 2. Composition by identity, on the left
        # Expect the original transformation
        rot_vec_2 = self.so3_group.random_uniform()
        mat_2 = self.so3_group.matrix_from_rotation_vector(rot_vec_2)

        result_2 = self.group.compose(self.group.identity, mat_2)
        expected_2 = mat_2

        norm = gs.linalg.norm(expected_2)
        atol = RTOL
        if norm != 0:
            atol = RTOL * norm
        self.assertTrue(gs.allclose(result_2, expected_2, atol=atol),
                        '\nresult:\n{}'
                        '\nexpected:\n{}'.format(result_2,
                                                 expected_2))

    def test_compose_and_inverse(self):
        # 1. Compose transformation by its inverse on the right
        # Expect the group identity
        rot_vec_1 = self.so3_group.random_uniform()
        mat_1 = self.so3_group.matrix_from_rotation_vector(rot_vec_1)
        inv_mat_1 = self.group.inverse(mat_1)

        result_1 = self.group.compose(mat_1, inv_mat_1)
        expected_1 = self.group.identity

        norm = gs.linalg.norm(expected_1)
        atol = RTOL
        if norm != 0:
            atol = RTOL * norm

        self.assertTrue(gs.allclose(result_1, expected_1, atol=atol),
                        '\nresult:\n{}'
                        '\nexpected:\n{}'.format(result_1, expected_1))

        # 2. Compose transformation by its inverse on the left
        # Expect the group identity
        rot_vec_2 = self.so3_group.random_uniform()
        mat_2 = self.so3_group.matrix_from_rotation_vector(rot_vec_2)
        inv_mat_2 = self.group.inverse(mat_2)

        result_2 = self.group.compose(inv_mat_2, mat_2)
        expected_2 = self.group.identity

        norm = gs.linalg.norm(expected_2)
        atol = RTOL
        if norm != 0:
            atol = RTOL * norm

        self.assertTrue(gs.allclose(result_2, expected_2, atol=atol))


if __name__ == '__main__':
        unittest.main()
