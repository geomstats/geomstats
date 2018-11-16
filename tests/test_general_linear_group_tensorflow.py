"""
Unit tests for General Linear group.
"""

import importlib
import os
import tensorflow as tf

import geomstats.backend as gs
import tests.helper as helper

from geomstats.general_linear_group import GeneralLinearGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

RTOL = 1e-5


class TestGeneralLinearGroupTensorFlow(tf.test.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        n = 3
        self.group = GeneralLinearGroup(n=n)
        # We generate invertible matrices using so3_group
        self.so3_group = SpecialOrthogonalGroup(n=n)

    @classmethod
    def setUpClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        importlib.reload(gs)

    @classmethod
    def tearDownClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'numpy'
        importlib.reload(gs)

    def test_belongs(self):
        """
        A rotation matrix belongs to the matrix Lie group
        of invertible matrices.
        """
        rot_vec = tf.convert_to_tensor([0.2, -0.1, 0.1])
        rot_mat = self.so3_group.matrix_from_rotation_vector(rot_vec)
        result = self.group.belongs(rot_mat)
        expected = tf.convert_to_tensor([True])

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_compose(self):
        # 1. Composition by identity, on the right
        # Expect the original transformation
        rot_vec = tf.convert_to_tensor([0.2, -0.1, 0.1])
        mat = self.so3_group.matrix_from_rotation_vector(rot_vec)

        result = self.group.compose(mat, self.group.identity)
        expected = mat
        expected = helper.to_matrix(mat)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        # 2. Composition by identity, on the left
        # Expect the original transformation
        rot_vec = tf.convert_to_tensor([0.2, 0.1, -0.1])
        mat = self.so3_group.matrix_from_rotation_vector(rot_vec)

        result = self.group.compose(self.group.identity, mat)
        expected = mat

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_inverse(self):
        mat = tf.convert_to_tensor([
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 10.]])
        result = self.group.inverse(mat)
        expected = 1. / 3. * tf.convert_to_tensor([
            [-2., -4., 3.],
            [-2., 11., -6.],
            [3., -6., 3.]])
        expected = helper.to_matrix(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_compose_and_inverse(self):
        # 1. Compose transformation by its inverse on the right
        # Expect the group identity
        rot_vec = tf.convert_to_tensor([0.2, 0.1, 0.1])
        mat = self.so3_group.matrix_from_rotation_vector(rot_vec)
        inv_mat = self.group.inverse(mat)

        result = self.group.compose(mat, inv_mat)
        expected = self.group.identity
        expected = helper.to_matrix(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        # 2. Compose transformation by its inverse on the left
        # Expect the group identity
        rot_vec = tf.convert_to_tensor([0.7, 0.1, 0.1])
        mat = self.so3_group.matrix_from_rotation_vector(rot_vec)
        inv_mat = self.group.inverse(mat)

        result = self.group.compose(inv_mat, mat)
        expected = self.group.identity
        expected = helper.to_matrix(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))


if __name__ == '__main__':
    tf.test.main()
