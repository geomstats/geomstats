"""Unit tests for matrix lie group module."""

import numpy as np
import unittest

from geomstats.matrix_lie_group import MatrixLieGroup

from geomstats.special_orthogonal_group import SpecialOrthogonalGroup


class TestMatrixLieGroupMethods(unittest.TestCase):
    N = 3
    GROUP = MatrixLieGroup(dimension=N ** 2,
                           n=N)
    # We generate invertible matrices using SO3
    SO3 = SpecialOrthogonalGroup(n=N)

    def test_belongs(self):
        """
        A rotation matrix belongs to the matrix Lie group
        of invertible matrices.
        """
        rot_vec = self.SO3.random_uniform()
        rot_mat = self.SO3.matrix_from_rotation_vector(rot_vec)

        self.assertTrue(self.GROUP.belongs(rot_mat))

    def test_compose(self):
        # 1. Composition by identity, on the right
        # Expect the original transformation
        rot_vec_1 = self.SO3.random_uniform()
        mat_1 = self.SO3.matrix_from_rotation_vector(rot_vec_1)

        result_1 = self.GROUP.compose(mat_1, self.GROUP.identity)
        expected_1 = mat_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Composition by identity, on the left
        # Expect the original transformation
        rot_vec_2 = self.SO3.random_uniform()
        mat_2 = self.SO3.matrix_from_rotation_vector(rot_vec_2)

        result_2 = self.GROUP.compose(self.GROUP.identity, mat_2)
        expected_2 = mat_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_compose_and_inverse(self):
        # 1. Compose transformation by its inverse on the right
        # Expect the group identity
        rot_vec_1 = self.SO3.random_uniform()
        mat_1 = self.SO3.matrix_from_rotation_vector(rot_vec_1)
        inv_mat_1 = self.GROUP.inverse(mat_1)

        result_1 = self.GROUP.compose(mat_1, inv_mat_1)
        expected_1 = self.GROUP.identity

        self.assertTrue(np.allclose(result_1, expected_1))

        # 2. Compose transformation by its inverse on the left
        # Expect the group identity
        rot_vec_2 = self.SO3.random_uniform()
        mat_2 = self.SO3.matrix_from_rotation_vector(rot_vec_2)
        inv_mat_2 = self.GROUP.inverse(mat_2)

        result_2 = self.GROUP.compose(inv_mat_2, mat_2)
        expected_2 = self.GROUP.identity

        self.assertTrue(np.allclose(result_2, expected_2))


if __name__ == '__main__':
        unittest.main()
