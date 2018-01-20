"""Unit tests for special orthogonal group module."""

import numpy as np
import unittest

from geomstats.special_orthogonal_group import SpecialOrthogonalGroup


class TestSpecialOrthogonalGroupMethods(unittest.TestCase):
    DIMENSION = 3
    GROUP = SpecialOrthogonalGroup(dimension=DIMENSION)
    METRIC = GROUP.bi_invariant_metric

    def test_regularize(self):
        rot_vec_0 = np.array([0., 0., 0.])
        rot_vec_0 = self.GROUP.regularize(rot_vec_0)
        rot_vec_0_expected = np.array([0., 0., 0.])
        self.assertTrue(np.allclose(rot_vec_0, rot_vec_0_expected))

        rot_vec_1 = 2.5 * np.pi * np.array([0., 0., 1.])
        rot_vec_1 = self.GROUP.regularize(rot_vec_1)
        rot_vec_1_expected = np.pi / 2. * np.array([0., 0., 1.])
        self.assertTrue(np.allclose(rot_vec_1, rot_vec_1_expected))

        rot_vec_2 = 1.5 * np.pi * np.array([0., 1., 0.])
        rot_vec_2 = self.GROUP.regularize(rot_vec_2)
        rot_vec_2_expected = np.pi / 2. * np.array([0., -1., 0.])
        self.assertTrue(np.allclose(rot_vec_2, rot_vec_2_expected))

        rot_vec_3 = 11 * np.pi * np.array([1., 2., 3.])
        rot_vec_3 = self.GROUP.regularize(rot_vec_3)
        # This factor comes from the norm of rot_vec_3 modulo pi
        fact = 0.84176874548664671 * np.pi / np.sqrt(14)
        rot_vec_3_expected = fact * np.array([-1., -2., -3.])
        self.assertTrue(np.allclose(rot_vec_3, rot_vec_3_expected))

        rot_vec_4 = 1e-15 * np.pi * np.array([1., 2., 3.])
        rot_vec_4 = self.GROUP.regularize(rot_vec_4)
        expected_rot_vec_4 = rot_vec_4
        self.assertTrue(np.allclose(rot_vec_4, expected_rot_vec_4))

        rot_vec_5 = 1e-11 * np.array([12., 1., -81.])
        rot_vec_5 = self.GROUP.regularize(rot_vec_5)
        expected_rot_vec_5 = rot_vec_5
        self.assertTrue(np.allclose(rot_vec_5, expected_rot_vec_5))

    def test_matrix_from_rotation_vector(self):
        rot_vec_0 = self.GROUP.identity
        rot_mat_0 = self.GROUP.matrix_from_rotation_vector(rot_vec_0)
        expected_rot_mat_0 = np.eye(3)
        self.assertTrue(np.allclose(rot_mat_0, expected_rot_mat_0))

        rot_vec_1 = np.array([np.pi / 3., 0., 0.])
        rot_mat_1 = self.GROUP.matrix_from_rotation_vector(rot_vec_1)
        expected_rot_mat_1 = np.array([[1., 0., 0.],
                                       [0., 0.5, -np.sqrt(3) / 2],
                                       [0., np.sqrt(3) / 2, 0.5]])
        self.assertTrue(np.allclose(rot_mat_1, expected_rot_mat_1))

        rot_vec_3 = 1e-11 * np.array([12., 1., -81.])
        angle = np.linalg.norm(rot_vec_3)
        skew_rot_vec_3 = 1e-11 * np.array([[0., 81., 1.],
                                           [-81., 0., -12.],
                                           [-1., 12., 0.]])
        coef_1 = np.sin(angle) / angle
        coef_2 = (1 - np.cos(angle)) / (angle ** 2)
        expected_rot_mat_3 = (np.identity(3)
                              + coef_1 * skew_rot_vec_3
                              + coef_2 * np.dot(skew_rot_vec_3,
                                                skew_rot_vec_3))
        rot_mat_3 = self.GROUP.matrix_from_rotation_vector(rot_vec_3)
        self.assertTrue(np.allclose(rot_mat_3, expected_rot_mat_3))

        rot_vec_6 = np.array([.1, 1.3, -.5])
        angle = np.linalg.norm(rot_vec_6)
        skew_rot_vec_6 = np.array([[0., .5, 1.3],
                                   [-.5, 0., -.1],
                                   [-1.3, .1, 0.]])

        coef_1 = np.sin(angle) / angle
        coef_2 = (1 - np.cos(angle)) / (angle ** 2)
        rot_mat_6 = self.GROUP.matrix_from_rotation_vector(rot_vec_6)
        expected_rot_mat_6 = (np.identity(3)
                              + coef_1 * skew_rot_vec_6
                              + coef_2 * np.dot(skew_rot_vec_6,
                                                skew_rot_vec_6))
        self.assertTrue(np.allclose(rot_mat_6, expected_rot_mat_6))

    def test_rotation_vector_from_matrix(self):
        rot_mat = np.array([[1., 0., 0.],
                            [0., np.cos(.12), -np.sin(.12)],
                            [0, np.sin(.12), np.cos(.12)]])
        rot_vec = self.GROUP.rotation_vector_from_matrix(rot_mat)
        expected_rot_vec = .12 * np.array([1., 0., 0.])

        self.assertTrue(np.allclose(rot_vec, expected_rot_vec))

    def test_rotation_vector_and_rotation_matrix(self):
        """
        This tests that the composition of
        rotation_vector_from_matrix
        and
        matrix_from_rotation_vector
        is the identity.
        """
        rot_vec_1 = np.array([np.pi / 3., 0., 0.])
        rot_mat_1 = self.GROUP.matrix_from_rotation_vector(rot_vec_1)
        result_rot_vec_1 = self.GROUP.rotation_vector_from_matrix(
                rot_mat_1)
        self.assertTrue(np.allclose(rot_vec_1, result_rot_vec_1))

        rot_vec_2 = 1e-11 * np.array([12., 1., -81.])
        angle = np.linalg.norm(rot_vec_2)
        skew_rot_vec_2 = 1e-11 * np.array([[0., 81., 1.],
                                           [-81., 0., -12.],
                                           [-1., 12., 0.]])
        coef_1 = np.sin(angle) / angle
        coef_2 = (1 - np.cos(angle)) / (angle ** 2)
        rot_mat_2 = (np.identity(3)
                     + coef_1 * skew_rot_vec_2
                     + coef_2 * np.dot(skew_rot_vec_2,
                                       skew_rot_vec_2))
        rot_vec_2 = self.GROUP.rotation_vector_from_matrix(rot_mat_2)
        result_rot_mat_2 = self.GROUP.matrix_from_rotation_vector(
                rot_vec_2)
        self.assertTrue(np.allclose(rot_mat_2, result_rot_mat_2))

        rot_vec_3 = np.array([0.1, 1.4, -0.5])
        rot_mat_3 = self.GROUP.matrix_from_rotation_vector(rot_vec_3)
        result_rot_vec_3 = self.GROUP.rotation_vector_from_matrix(
                rot_mat_3)
        self.assertTrue(np.allclose(rot_vec_3, result_rot_vec_3))

        rot_vec_4 = np.array([0., 2., 1.])
        rot_mat_4 = self.GROUP.matrix_from_rotation_vector(rot_vec_4)
        result_rot_vec_4 = self.GROUP.rotation_vector_from_matrix(
                rot_mat_4)
        self.assertTrue(np.allclose(rot_vec_4, result_rot_vec_4))

        rot_vec_5 = self.GROUP.identity
        rot_mat_5 = self.GROUP.matrix_from_rotation_vector(rot_vec_5)
        result_rot_vec_5 = self.GROUP.rotation_vector_from_matrix(
                rot_mat_5)
        self.assertTrue(np.allclose(rot_vec_5, result_rot_vec_5))

    def test_exp(self):
        """
        The Riemannian exp and log are inverse functions of each other.
        This test is the inverse of test_log's.
        """
        theta = np.pi / 5
        rot_vec_base_point = theta / np.sqrt(3.) * np.array([1., 1., 1.])
        # Note: the rotation vector for the reference point
        # needs to be regularized.

        # 1: Exponential of 0 gives the reference point
        rot_vec_1 = np.array([0, 0, 0])
        expected_1 = rot_vec_base_point

        exp_1 = self.METRIC.exp(base_point=rot_vec_base_point,
                                tangent_vec=rot_vec_1)
        self.assertTrue(np.allclose(exp_1, expected_1))

        # 2: General case - computed manually
        rot_vec_2 = np.pi / 4 * np.array([1, 0, 0])
        phi = (np.pi / 10) / (np.tan(np.pi / 10))
        skew = np.array([[0., -1., 1.],
                         [1., 0., -1.],
                         [-1., 1., 0.]])
        jacobian = (phi * np.eye(3)
                    + (1 - phi) / 3 * np.ones([3, 3])
                    + np.pi / (10 * np.sqrt(3)) * skew)
        inv_jacobian = np.linalg.inv(jacobian)
        expected_2 = self.GROUP.compose(rot_vec_base_point,
                                        np.dot(inv_jacobian, rot_vec_2))

        exp_2 = self.METRIC.exp(base_point=rot_vec_base_point,
                                tangent_vec=rot_vec_2)
        self.assertTrue(np.allclose(exp_2, expected_2))

    def test_log(self):
        """
        The Riemannian exp and log are inverse functions of each other.
        This test is the inverse of test_exp's.
        """
        theta = np.pi / 5.
        rot_vec_base_point = theta / np.sqrt(3.) * np.array([1., 1., 1.])
        # Note: the rotation vector for the reference point
        # needs to be regularized.

        # The Logarithm of a point at itself gives 0.
        rot_vec_1 = rot_vec_base_point
        expected_1 = np.array([0, 0, 0])
        log_1 = self.METRIC.log(base_point=rot_vec_base_point,
                                point=rot_vec_1)
        self.assertTrue(np.allclose(log_1, expected_1))

        # General case: this is the inverse test of test 1 for riemannian exp
        expected_2 = np.pi / 4 * np.array([1, 0, 0])
        phi = (np.pi / 10) / (np.tan(np.pi / 10))
        skew = np.array([[0., -1., 1.],
                         [1., 0., -1.],
                         [-1., 1., 0.]])
        jacobian = (phi * np.eye(3)
                    + (1 - phi) / 3 * np.ones([3, 3])
                    + np.pi / (10 * np.sqrt(3)) * skew)
        inv_jacobian = np.linalg.inv(jacobian)
        rot_vec_2 = self.GROUP.compose(rot_vec_base_point,
                                       np.dot(inv_jacobian, expected_2))

        log_2 = self.METRIC.log(base_point=rot_vec_base_point,
                                point=rot_vec_2)
        self.assertTrue(np.allclose(log_2, expected_2))

    def test_log_and_exp(self):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        theta = 12. * np.pi / 5.
        rot_vec_base_point = theta / np.sqrt(3.) * np.array([1., 1., 1.])

        rot_vec_1 = np.array([0, 0, 0])
        aux_1 = self.METRIC.exp(base_point=rot_vec_base_point,
                                tangent_vec=rot_vec_1)
        result_1 = self.METRIC.log(base_point=rot_vec_base_point,
                                   point=aux_1)

        self.assertTrue(np.allclose(result_1, rot_vec_1))

        rot_vec_2 = np.pi / (2 * np.sqrt(3)) * np.array([1, 0, 0])

        aux_2 = self.METRIC.exp(base_point=rot_vec_base_point,
                                tangent_vec=rot_vec_2)
        result_2 = self.METRIC.log(base_point=rot_vec_base_point,
                                   point=aux_2)

        self.assertTrue(np.allclose(result_2, rot_vec_2))


if __name__ == '__main__':
        unittest.main()
