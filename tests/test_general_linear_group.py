"""
Unit tests for General Linear group.
"""

import warnings

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper

from geomstats.general_linear_group import GeneralLinearGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

RTOL = 1e-5


class TestGeneralLinearGroupMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        self.n = 3
        self.n_samples = 2
        self.group = GeneralLinearGroup(n=self.n)
        # We generate invertible matrices using so3_group
        self.so3_group = SpecialOrthogonalGroup(n=self.n)

        warnings.simplefilter('ignore', category=ImportWarning)

    @geomstats.tests.np_only
    def test_belongs(self):
        """
        A rotation matrix belongs to the matrix Lie group
        of invertible matrices.
        """
        rot_vec = gs.array([0.2, -0.1, 0.1])
        rot_mat = self.so3_group.matrix_from_rotation_vector(rot_vec)
        result = self.group.belongs(rot_mat)
        expected = gs.array([True])

        self.assertAllClose(result, expected)

    def test_compose(self):
        # 1. Composition by identity, on the right
        # Expect the original transformation
        rot_vec = gs.array([0.2, -0.1, 0.1])
        mat = self.so3_group.matrix_from_rotation_vector(rot_vec)

        result = self.group.compose(mat, self.group.identity)
        expected = mat
        expected = helper.to_matrix(mat)

        self.assertAllClose(result, expected)

        # 2. Composition by identity, on the left
        # Expect the original transformation
        rot_vec = gs.array([0.2, 0.1, -0.1])
        mat = self.so3_group.matrix_from_rotation_vector(rot_vec)

        result = self.group.compose(self.group.identity, mat)
        expected = mat

        self.assertAllClose(result, expected)

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

        self.assertAllClose(result, expected)

    def test_compose_and_inverse(self):
        # 1. Compose transformation by its inverse on the right
        # Expect the group identity
        rot_vec = gs.array([0.2, 0.1, 0.1])
        mat = self.so3_group.matrix_from_rotation_vector(rot_vec)
        inv_mat = self.group.inverse(mat)

        result = self.group.compose(mat, inv_mat)
        expected = self.group.identity
        expected = helper.to_matrix(expected)

        self.assertAllClose(result, expected)

        # 2. Compose transformation by its inverse on the left
        # Expect the group identity
        rot_vec = gs.array([0.7, 0.1, 0.1])
        mat = self.so3_group.matrix_from_rotation_vector(rot_vec)
        inv_mat = self.group.inverse(mat)

        result = self.group.compose(inv_mat, mat)
        expected = self.group.identity
        expected = helper.to_matrix(expected)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_group_log_and_exp(self):
        point = 5 * gs.eye(self.n)

        group_log = self.group.group_log(point)
        result = self.group.group_exp(group_log)
        expected = point
        expected = helper.to_matrix(expected)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
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

        self.assertAllClose(result, expected, rtol=1e-3)

    @geomstats.tests.np_and_tf_only
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

        self.assertAllClose(result, expected, atol=1e-4)

    @geomstats.tests.np_and_tf_only
    def test_expm_and_logm_vectorization_symmetric(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])
        result = self.group.group_exp(self.group.group_log(point))
        expected = point

        self.assertAllClose(result, expected)


if __name__ == '__main__':
        geomstats.tests.main()
