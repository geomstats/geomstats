"""Unit tests to check the maths functions."""

import numpy as np
import unittest

import rotations
import rigid_transformations


class TestGeomstatsMethods(unittest.TestCase):

    def test_rotation_matrix_from_rotation_vector(self):
        rotation_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [-0.1, 0.2, 0]]
        for rot_vec in rotation_vectors:
            R = rotations.rotation_matrix_from_rotation_vector(rot_vec)
            R2 = rotations.rotation_matrix_from_rotation_vector2(rot_vec)
            self.assertTrue(np.allclose(R, R2))

            rot_vec_test = rotations.rotation_vector_from_rotation_matrix(R)
            self.assertTrue(np.allclose(rot_vec, rot_vec_test))

            rot_vec_test2 = rotations.rotation_vector_from_rotation_matrix(R2)
            self.assertTrue(np.allclose(rot_vec, rot_vec_test2))

    def test_rotation_vector_from_rotation_matrix(self):
        rotation_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [-0.1, 0.2, 0]]
        for rot_vec in rotation_vectors:
            R = rotations.rotation_matrix_from_rotation_vector(rot_vec)

            rot_vec_test = rotations.rotation_vector_from_rotation_matrix(R)
            self.assertTrue(np.allclose(rot_vec, rot_vec_test))

    def test_regularize_rotation_vector(self):
        u1 = 2.5 * np.pi * np.array([0., 0., 1.])
        r1 = rotations.regularize_rotation_vector(u1)
        r1_expected = np.pi / 2. * np.array([0., 0., 1.])
        self.assertTrue(np.allclose(r1, r1_expected))

        u2 = 1.5 * np.pi * np.array([0., 1., 0.])
        r2 = rotations.regularize_rotation_vector(u2)
        r2_expected = np.pi / 2. * np.array([0., -1., 0.])
        self.assertTrue(np.allclose(r2, r2_expected))

        u3 = 11 * np.pi * np.array([1., 2., 3.])
        r3 = rotations.regularize_rotation_vector(u3)
        fact = 0.84176874548664671 * np.pi / np.sqrt(14)
        r3_expected = fact*np.array([-1., -2., -3.])
        self.assertTrue(np.allclose(r3, r3_expected))

        u4 = 1e-15 * np.pi * np.array([1., 2., 3.])
        r4 = rotations.regularize_rotation_vector(u4)
        r4_expected = u4
        self.assertTrue(np.allclose(r4, r4_expected))

        u5 = 1e-11 * np.array([12., 1., -81.])
        r5 = rotations.regularize_rotation_vector(u5)
        r5_expected = u5
        self.assertTrue(np.allclose(r5, r5_expected))

    def test_representations_of_rotations(self):
        r1 = np.array([np.pi / 3., 0., 0.])
        R1 = rotations.rotation_matrix_from_rotation_vector(r1)

        self.assertTrue(np.allclose(
            r1,
            rotations.rotation_vector_from_rotation_matrix(R1)))

        r2 = 12 * np.pi / (5. * np.sqrt(3.)) * np.array([1., 1., 1.])
        r2 = rotations.regularize_rotation_vector(r2)
        R2 = rotations.rotation_matrix_from_rotation_vector(r2)
        self.assertTrue(np.allclose(
            r2,
            rotations.rotation_vector_from_rotation_matrix(R2)))

        r3 = 1e-11 * np.array([12., 1., -81.])
        theta = np.linalg.norm(r3)

        r3 = 1e-11 * np.array([12., 1., -81.])
        theta = np.linalg.norm(r3)
        K = 1e-11 * np.array([[0., 81., 1.],
                              [-81., 0., -12.],
                              [-1., 12., 0.]])
        c1 = np.sin(theta) / theta
        c2 = (1 - np.cos(theta)) / (theta ** 2)
        R3_expected = np.identity(3) + c1 * K + c2 * np.dot(K, K)
        R3 = rotations.rotation_matrix_from_rotation_vector(r3)

        self.assertTrue(np.allclose(R3, R3_expected))

        R4 = np.array([[1., 0., 0.],
                       [0., np.cos(.12), -np.sin(.12)],
                       [0, np.sin(.12), np.cos(.12)]])
        r4 = rotations.rotation_vector_from_rotation_matrix(R4)
        r4_expected = .12 * np.array([1., 0., 0.])

        self.assertTrue(np.allclose(r4, r4_expected))
        self.assertTrue(np.allclose(
            R4,
            rotations.rotation_matrix_from_rotation_vector(r4)))

        R5 = np.array([[1., 0., 0.],
                       [0., np.cos(1e-14), -np.sin(1e-14)],
                       [0, np.sin(1e-14), np.cos(1e-14)]])
        r5 = rotations.rotation_vector_from_rotation_matrix(R5)
        r5_expected = 1e-14 * np.array([1., 0., 0.])

        self.assertTrue(np.allclose(r5, r5_expected))
        self.assertTrue(np.allclose(
            R5,
            rotations.rotation_matrix_from_rotation_vector(r5)))

        r6 = np.array([.1, 1.3, -.5])
        theta = np.linalg.norm(r6)
        K = np.array([[0., .5, 1.3],
                      [-.5, 0., -.1],
                      [-1.3, .1, 0.]])

        c1 = np.sin(theta) / theta
        c2 = (1 - np.cos(theta)) / (theta ** 2)
        R6_expected = np.identity(3) + c1 * K + c2 * np.dot(K, K)
        R6 = rotations.rotation_matrix_from_rotation_vector(r6)
        self.assertTrue(np.allclose(R6, R6_expected))

    def test_rigid_transformations_group_exp_log(self):
        x1 = np.array([1, 0, -3])
        r1 = np.pi / (3 * np.sqrt(2)) * np.array([0, 0, 0])

        x2 = np.array([4, 0, 0])
        r2 = np.pi / (2 * np.sqrt(3)) * np.array([1, 0, 0])

        x3 = np.array([1.2, -3.6, 50])
        r3 = np.pi / (2 * np.sqrt(3)) * np.array([1, -20, 50])

        x4 = np.array([4, 10, -2])
        r4 = (np.pi / (2 * np.sqrt(3)) *
              np.array([6 * 1e-8, 5.5 * 1e-7, -2 * 1e-6]))

        xs = [x1, x2, x3, x4]
        rs = map(rotations.regularize_rotation_vector, [r1, r2, r3, r4])
        for (x, r) in zip(xs, rs):
            (u, a) = rigid_transformations.group_log(x, r)
            (x_result, r_result) = rigid_transformations.group_exp(u, a)
            (x_expected, r_expected) = (x, r)

            self.assertTrue(np.allclose(x_result, x_expected))
            self.assertTrue(np.allclose(r_result, r_expected))

    def test_rotations_riemannian_exp_log(self):
        r_ref = rotations.regularize_rotation_vector(np.array([-1, 3, 6]))
        r1 = np.pi / (3 * np.sqrt(2)) * np.array([0, 0, 0])
        r2 = np.pi / (2 * np.sqrt(3)) * np.array([1, 0, 0])
        r3 = np.pi / (2 * np.sqrt(3)) * np.array([1, -20, 50])
        r4 = (np.pi / (2 * np.sqrt(3)) *
              np.array([6 * 1e-8, 5.5 * 1e-7, -2 * 1e-6]))

        rs = map(rotations.regularize_rotation_vector, [r1, r2, r3, r4])
        for r in rs:
            u = rotations.riemmanian_log(r_ref, r)
            r_result = rotations.riemannian_exp(r_ref, u)
            r_expected = r
            self.assertTrue(np.allclose(r_result, r_expected))

    def test_rigid_transformations_riemannian_exp_log(self):
        x_ref = np.array([1, 2, 3])
        r_ref = rotations.regularize_rotation_vector(np.array([-1, 3, 6]))

        x1 = np.array([1, 0, -3])
        r1 = np.pi / (3 * np.sqrt(2)) * np.array([0, 0, 0])

        x2 = np.array([4, 0, 0])
        r2 = np.pi / (2 * np.sqrt(3)) * np.array([1, 0, 0])

        x3 = np.array([1.2, -3.6, 50])
        r3 = np.pi / (2 * np.sqrt(3)) * np.array([1, -20, 50])

        x4 = np.array([4, 10, -2])
        r4 = (np.pi / (2 * np.sqrt(3)) *
              np.array([6 * 1e-8, 5.5 * 1e-7, -2 * 1e-6]))

        xs = [x1, x2, x3, x4]
        rs = map(rotations.regularize_rotation_vector, [r1, r2, r3, r4])
        for (x, r) in zip(xs, rs):
            (u, a) = rigid_transformations.riemannian_log(x_ref, r_ref, x, r)
            (x_result, r_result) = rigid_transformations.riemannian_exp(
                    x_ref, r_ref, u, a)
            (x_expected, r_expected) = (x, r)

            self.assertTrue(np.allclose(x_result, x_expected))
            self.assertTrue(np.allclose(r_result, r_expected))

if __name__ == '__main__':
        unittest.main()
