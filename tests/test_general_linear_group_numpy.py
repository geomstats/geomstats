"""
Unit tests for General Linear group.
"""

import unittest
import warnings

import geomstats.backend as gs
import tests.helper as helper

from geomstats.general_linear_group import GeneralLinearGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

RTOL = 1e-5


class TestGeneralLinearGroupMethods(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        self.n = 3
        self.n_samples = 2
        self.group = GeneralLinearGroup(n=self.n)
        # We generate invertible matrices using so3_group
        self.so3_group = SpecialOrthogonalGroup(n=self.n)

        warnings.simplefilter('ignore', category=ImportWarning)

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

    def test_inverse(self):
        mat = gs.array([
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 10.]])
        result = self.group.inverse(mat)
        expected = 1. / 3. * gs.array([
            [-2., -4., 3.],
            [-2., 11., -6.],
            [3., -6., 3.]])
        expected = helper.to_matrix(expected)
        self.assertTrue(gs.allclose(result, expected))

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

    def test_group_log_and_exp(self):
        point_1 = 5 * gs.eye(4)
        group_log_1 = self.group.group_log(point_1)
        result_1 = self.group.group_exp(group_log_1)
        expected_1 = point_1

        self.assertTrue(gs.allclose(result_1, expected_1))

    def test_group_exp_vectorization(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])

        expected = gs.array([[[7.38905609, 0., 0.],
                              [0., 20.0855369, 0.],
                              [0., 0., 54.5981500]],
                             [[2.718281828, 0., 0.],
                              [0., 148.413159, 0.],
                              [0., 0., 403.42879349]]])

        result = self.group.group_exp(point)

        self.assertTrue(gs.allclose(result, expected))

    def test_group_log_vectorization(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])

        expected = gs.array([[[0.693147180, 0., 0.],
                              [0., 1.09861228866, 0.],
                              [0., 0., 1.38629436]],
                             [[0., 0., 0.],
                              [0., 1.609437912, 0.],
                              [0., 0., 1.79175946]]])

        result = self.group.group_log(point)

        self.assertTrue(gs.allclose(result, expected))

    def test_expm_and_logm_vectorization_symmetric(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])
        result = self.group.group_exp(self.group.group_log(point))
        expected = point

        self.assertTrue(gs.allclose(result, expected))

    def test_expm_and_logm_vectorization_random_rotations(self):
        point = self.so3_group.random_uniform(self.n_samples)
        point = self.so3_group.matrix_from_rotation_vector(point)
        result = self.group.group_log(self.group.group_exp(point))
        expected = point

        self.assertTrue(gs.allclose(result, expected))


if __name__ == '__main__':
        unittest.main()
