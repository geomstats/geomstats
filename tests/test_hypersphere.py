"""Unit tests for hypersphere module."""

from geomstats.hypersphere import HypersphereMetric
from geomstats.hypersphere import Hypersphere

import numpy as np
import unittest


class TestHypersphereMethods(unittest.TestCase):
    DIMENSION = 5
    METRIC = HypersphereMetric()
    SPACE = Hypersphere(dimension=DIMENSION)

    def test_riemannian_log_and_exp(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.

        NB: points on the n-dimensional sphere are
        (n+1)-D vectors of norm 1.
        """
        # Riemannian Log then Riemannian Exp
        # General case
        ref_point_1 = np.array([1., 2., 3., 4., 6.])
        ref_point_1 = ref_point_1 / np.linalg.norm(ref_point_1)
        point_1 = np.array([0., 5., 6., 2., -1])
        point_1 = point_1 / np.linalg.norm(point_1)

        riem_log_1 = self.METRIC.riemannian_log(ref_point_1, point_1)
        result_1 = self.METRIC.riemannian_exp(ref_point_1, riem_log_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # Riemannian Log then Riemannian Exp
        # Edge case: two very close points, ref_point_2 and point_2,
        # form an angle < epsilon
        ref_point_2 = np.array([1., 2., 3., 4., 6.])
        ref_point_2 = ref_point_2 / np.linalg.norm(ref_point_2)
        point_2 = ref_point_2 + 1e-12 * np.array([-1., -2., 1., 1., .1])
        point_2 = point_2 / np.linalg.norm(point_2)

        riem_log_2 = self.METRIC.riemannian_log(ref_point_2, point_2)
        result_2 = self.METRIC.riemannian_exp(ref_point_2, riem_log_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_riemannian_exp_and_log_and_projection_to_tangent_space(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.

        NB: points on the n-dimensional sphere are
        (n+1)-D vectors of norm 1.
        """
        # Riemannian Exp then Riemannian Log
        # General case
        # NB: Riemannian log gives a regularized tangent vector,
        # so we take the norm modulo 2 * pi.
        ref_point_1 = np.array([7., -3., 8., 33., 45., -129, 0.])
        ref_point_1 = ref_point_1 / np.linalg.norm(ref_point_1)
        vector_1 = np.array([9., 54., 0., 0., -1., -33., 2.])
        vector_1 = self.SPACE.projection_to_tangent_space(ref_point_1,
                                                          vector_1)

        riem_exp_1 = self.METRIC.riemannian_exp(ref_point_1, vector_1)
        result_1 = self.METRIC.riemannian_log(ref_point_1, riem_exp_1)

        expected_1 = self.SPACE.projection_to_tangent_space(ref_point_1,
                                                            vector_1)
        norm_expected_1 = np.linalg.norm(expected_1)
        regularized_norm_expected_1 = np.mod(norm_expected_1, 2 * np.pi)
        expected_1 = expected_1 / norm_expected_1 * regularized_norm_expected_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # Riemannian Exp then Riemannian Log
        # Edge case: tangent vector has norm < epsilon
        ref_point_2 = np.array([10., -2., -.5, 34.])
        ref_point_2 = ref_point_2 / np.linalg.norm(ref_point_2)
        vector_2 = 1e-10 * np.array([.06, -51., 6., 5.])
        vector_2 = self.SPACE.projection_to_tangent_space(ref_point_2,
                                                          vector_2)

        riem_exp_2 = self.METRIC.riemannian_exp(ref_point_2, vector_2)
        result_2 = self.METRIC.riemannian_log(ref_point_2, riem_exp_2)
        expected_2 = self.SPACE.projection_to_tangent_space(ref_point_2,
                                                            vector_2)

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_riemannian_dist(self):
        # Distance between a point and itself is 0.
        point_a_1 = np.array([10., -2., -.5])
        point_b_1 = point_a_1
        result_1 = self.METRIC.riemannian_dist(point_a_1, point_b_1)
        expected_1 = 0.

        self.assertTrue(np.allclose(result_1, expected_1))

        # Distance between two orthogonal points is pi / 2.
        point_a_2 = np.array([10., -2., -.5])
        point_b_2 = np.array([2., 10, 0.])
        assert np.dot(point_a_2, point_b_2) == 0

        result_2 = self.METRIC.riemannian_dist(point_a_2, point_b_2)
        expected_2 = np.pi / 2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_riemannian_exp_and_dist_and_projection_to_tangent_space(self):
        ref_point_1 = np.array([16., -2., -2.5, 84.])
        ref_point_1 = ref_point_1 / np.linalg.norm(ref_point_1)

        vector_1 = np.array([9., 0., -1., -2.])
        tangent_vec_1 = self.SPACE.projection_to_tangent_space(ref_point_1,
                                                               vector_1)
        riem_exp_1 = self.METRIC.riemannian_exp(ref_point_1, tangent_vec_1)

        result_1 = self.METRIC.riemannian_dist(ref_point_1, riem_exp_1)
        expected_1 = np.mod(np.linalg.norm(tangent_vec_1), 2 * np.pi)

        self.assertTrue(np.allclose(result_1, expected_1))


if __name__ == '__main__':
        unittest.main()
