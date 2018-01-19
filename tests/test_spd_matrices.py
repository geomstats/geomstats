"""Unit tests for symmetric positive definite matrices."""

import numpy as np
import unittest

import geomstats.spd_matrices_space as spd_matrices_space
from geomstats.spd_matrices_space import SPDMatricesSpace


class TestSPDMatricesSpaceMethods(unittest.TestCase):
    DIMENSION = 3
    SPACE = SPDMatricesSpace(dimension=DIMENSION)
    METRIC = SPACE.riemannian_metric

    def test_is_symmetric(self):
        sym_mat = np.array([[1, 2],
                            [2, 1]])
        self.assertTrue(spd_matrices_space.is_symmetric(sym_mat))

        not_a_sym_mat = np.array([[1., 0.6, -3.],
                                  [6., -7., 0.],
                                  [0., 7., 8.]])
        self.assertFalse(spd_matrices_space.is_symmetric(not_a_sym_mat))

    def test_make_symmetric(self):
        sym_mat = np.array([[1, 2],
                            [2, 1]])
        result = spd_matrices_space.make_symmetric(sym_mat)
        expected = sym_mat
        self.assertTrue(np.allclose(result, expected))

        mat = np.array([[1, 2, 3],
                        [0, 0, 0],
                        [3, 1, 1]])
        result = spd_matrices_space.make_symmetric(mat)
        expected = np.array([[1, 1, 3],
                            [1, 0, 0.5],
                            [3, 0.5, 1]])
        self.assertTrue(np.allclose(result, expected))

    def matrix_to_vector_and_vector_to_matrix(self):
        sym_mat_1 = np.array([[1., 0.6, -3.],
                              [0.6, 7., 0.],
                              [-3., 0., 8.]])
        vector_1 = self.SPACE.matrix_to_vector(sym_mat_1)
        result_1 = self.SPACE.vector_to_matrix(vector_1)
        expected_1 = sym_mat_1

        self.assertTrue(np.allclose(result_1, expected_1))

        vector_2 = np.array([1, 2, 3, 4, 5, 6])
        sym_mat_2 = self.SPACE.vector_to_matrix(vector_2)
        result_2 = self.SPACE.matrix_to_vector(sym_mat_2)
        expected_2 = vector_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_group_log_and_exp(self):
        point_1 = 5 * np.eye(4)
        group_log_1 = spd_matrices_space.group_log(point_1)
        result_1 = spd_matrices_space.group_exp(group_log_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_riemannian_log_and_exp(self):
        ref_point_1 = np.array([[5., 0., 0.],
                                [0., 7., 2.],
                                [0., 2., 8.]])
        point_1 = np.array([[9., 0., 0.],
                            [0., 5., 0.],
                            [0., 0., 1.]])

        riem_log_1 = self.METRIC.riemannian_log(ref_point_1, point_1)
        result_1 = self.METRIC.riemannian_exp(ref_point_1, riem_log_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))


if __name__ == '__main__':
        unittest.main()
