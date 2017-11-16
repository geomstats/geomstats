"""Unit tests for rotations module."""

import numpy as np
import unittest

import rotations


class TestRotationsMethods(unittest.TestCase):

    def test_regularize_rotation_vector(self):
        rot_vec_1 = 2.5 * np.pi * np.array([0., 0., 1.])
        rot_vec_1 = rotations.regularize_rotation_vector(rot_vec_1)
        rot_vec_1_expected = np.pi / 2. * np.array([0., 0., 1.])
        self.assertTrue(np.allclose(rot_vec_1, rot_vec_1_expected))

        rot_vec_2 = 1.5 * np.pi * np.array([0., 1., 0.])
        rot_vec_2 = rotations.regularize_rotation_vector(rot_vec_2)
        rot_vec_2_expected = np.pi / 2. * np.array([0., -1., 0.])
        self.assertTrue(np.allclose(rot_vec_2, rot_vec_2_expected))

        rot_vec_3 = 11 * np.pi * np.array([1., 2., 3.])
        rot_vec_3 = rotations.regularize_rotation_vector(rot_vec_3)
        fact = 0.84176874548664671 * np.pi / np.sqrt(14)
        rot_vec_3_expected = fact*np.array([-1., -2., -3.])
        self.assertTrue(np.allclose(rot_vec_3, rot_vec_3_expected))

        rot_vec_4 = 1e-15 * np.pi * np.array([1., 2., 3.])
        rot_vec_4 = rotations.regularize_rotation_vector(rot_vec_4)
        rot_vec_4_expected = rot_vec_4
        self.assertTrue(np.allclose(rot_vec_4, rot_vec_4_expected))

        rot_vec_5 = 1e-11 * np.array([12., 1., -81.])
        rot_vec_5 = rotations.regularize_rotation_vector(rot_vec_5)
        rot_vec_5_expected = rot_vec_5
        self.assertTrue(np.allclose(rot_vec_5, rot_vec_5_expected))

    def test_representations_of_rotations(self):
        rot_vec_1 = np.array([np.pi / 3., 0., 0.])
        rot_mat_1 = rotations.rotation_matrix_from_rotation_vector(rot_vec_1)
        rot_vec_1_test = rotations.rotation_vector_from_rotation_matrix(
                                                               rot_mat_1)
        self.assertTrue(np.allclose(rot_vec_1, rot_vec_1_test))

        rot_vec_2 = 12 * np.pi / (5. * np.sqrt(3.)) * np.array([1., 1., 1.])
        rot_mat_2 = rotations.rotation_matrix_from_rotation_vector(rot_vec_2)
        rot_vec_2_test = rotations.rotation_vector_from_rotation_matrix(
                                                               rot_mat_2)
        self.assertTrue(np.allclose(
                         rotations.regularize_rotation_vector(rot_vec_2),
                         rot_vec_2_test))

        rot_vec_3 = 1e-11 * np.array([12., 1., -81.])
        angle = np.linalg.norm(rot_vec_3)
        skew_rot_vec_3 = 1e-11 * np.array([[0., 81., 1.],
                                           [-81., 0., -12.],
                                           [-1., 12., 0.]])
        coef_1 = np.sin(angle) / angle
        coef_2 = (1 - np.cos(angle)) / (angle ** 2)
        rot_mat_3_expected = (np.identity(3)
                              + coef_1 * skew_rot_vec_3
                              + coef_2 * np.dot(skew_rot_vec_3,
                                                skew_rot_vec_3))
        rot_mat_3 = rotations.rotation_matrix_from_rotation_vector(rot_vec_3)
        self.assertTrue(np.allclose(rot_mat_3, rot_mat_3_expected))

        rot_mat_4 = np.array([[1., 0., 0.],
                              [0., np.cos(.12), -np.sin(.12)],
                              [0, np.sin(.12), np.cos(.12)]])
        rot_vec_4 = rotations.rotation_vector_from_rotation_matrix(rot_mat_4)
        rot_vec_4_expected = .12 * np.array([1., 0., 0.])
        rot_mat_4_expected = rotations.rotation_matrix_from_rotation_vector(
                                                                   rot_vec_4)

        self.assertTrue(np.allclose(rot_vec_4, rot_vec_4_expected))
        self.assertTrue(np.allclose(rot_mat_4, rot_mat_4_expected))

        rot_mat_5 = np.array([[1., 0., 0.],
                              [0., np.cos(1e-14), -np.sin(1e-14)],
                              [0., np.sin(1e-14), np.cos(1e-14)]])
        rot_vec_5 = rotations.rotation_vector_from_rotation_matrix(rot_mat_5)
        rot_vec_5_expected = 1e-14 * np.array([1., 0., 0.])
        rot_mat_5_expected = rotations.rotation_matrix_from_rotation_vector(
                                                                   rot_vec_5)

        self.assertTrue(np.allclose(rot_vec_5, rot_vec_5_expected))
        self.assertTrue(np.allclose(rot_mat_5, rot_mat_5_expected))

        rot_vec_6 = np.array([.1, 1.3, -.5])
        angle = np.linalg.norm(rot_vec_6)
        skew_rot_vec_6 = np.array([[0., .5, 1.3],
                                   [-.5, 0., -.1],
                                   [-1.3, .1, 0.]])

        coef_1 = np.sin(angle) / angle
        coef_2 = (1 - np.cos(angle)) / (angle ** 2)
        rot_mat_6 = rotations.rotation_matrix_from_rotation_vector(rot_vec_6)
        rot_mat_6_expected = (np.identity(3)
                              + coef_1 * skew_rot_vec_6
                              + coef_2 * np.dot(skew_rot_vec_6,
                                                skew_rot_vec_6))
        self.assertTrue(np.allclose(rot_mat_6, rot_mat_6_expected))

    def test_rotations_riemannian_exp_log(self):
        rot_vec_ref_point = np.array([-1, 3, 6])

        rot_vec_1 = np.pi / (3 * np.sqrt(2)) * np.array([0, 0, 0])
        rot_vec_2 = np.pi / (2 * np.sqrt(3)) * np.array([1, 0, 0])
        rot_vec_3 = np.pi / (2 * np.sqrt(3)) * np.array([1, -20, 50])
        rot_vec_4 = (np.pi / (2 * np.sqrt(3)) *
                     np.array([6 * 1e-8, 5.5 * 1e-7, -2 * 1e-6]))
        all_rot_vecs = [rot_vec_1, rot_vec_2, rot_vec_3, rot_vec_4]

        for rot_vec in all_rot_vecs:
            riem_log = rotations.riemannian_log(rot_vec,
                                                ref_point=rot_vec_ref_point)
            rot_vec_result = rotations.riemannian_exp(
                                              riem_log,
                                              ref_point=rot_vec_ref_point)
            rot_vec_expected = rotations.regularize_rotation_vector(rot_vec)
            self.assertTrue(np.allclose(rot_vec_result, rot_vec_expected))


if __name__ == '__main__':
        unittest.main()
