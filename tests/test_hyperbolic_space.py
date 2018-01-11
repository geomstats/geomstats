"""Unit tests for hyperbolic_space module."""

import geomstats.hyperbolic_space as hyperbolic_space

import numpy as np
import math
import unittest


class TestHyperbolicSpaceMethods(unittest.TestCase):
    def test_riemannian_log_and_exp(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        # General case
        ref_point_1 = hyperbolic_space.random_uniform(6, 1)
        point_1 = hyperbolic_space.random_uniform(6, 1)

        riem_log_1 = hyperbolic_space.riemannian_log(ref_point_1, point_1)
        result_1 = hyperbolic_space.riemannian_exp(ref_point_1, riem_log_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # Riemannian Log then Riemannian Exp
        # Edge case: two very close points, ref_point_2 and point_2,
        # form an angle < epsilon
        ref_point_intrinsic_2 = np.array([1., 2., 3., 4])
        ref_point_2 = hyperbolic_space.intrinsic_to_extrinsic_coords(
                                                       ref_point_intrinsic_2)
        point_intrinsic_2 = (ref_point_intrinsic_2
                             + 1e-12 * np.array([-1., -2., 1., 1.]))
        point_2 = hyperbolic_space.intrinsic_to_extrinsic_coords(
                                                       point_intrinsic_2)

        riem_log_2 = hyperbolic_space.riemannian_log(ref_point_2, point_2)
        result_2 = hyperbolic_space.riemannian_exp(ref_point_2, riem_log_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_riemannian_exp_and_log_and_projection_to_tangent_space(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Exp then Riemannian Log
        # General case
        ref_point_1 = hyperbolic_space.random_uniform(6, 1)
        # TODO(nina): this fails for high euclidean norms of vector_1
        vector_1 = np.array([9., 4., 0., 0., -1., -3., 2.])

        riem_exp_1 = hyperbolic_space.riemannian_exp(ref_point_1, vector_1)
        result_1 = hyperbolic_space.riemannian_log(ref_point_1, riem_exp_1)

        expected_1 = hyperbolic_space.projection_to_tangent_space(ref_point_1,
                                                                  vector_1)

        self.assertTrue(np.allclose(result_1, expected_1))

        # Riemannian Exp then Riemannian Log
        # Edge case: tangent vector has norm < epsilon
        ref_point_2 = hyperbolic_space.random_uniform(3, 1)
        vector_2 = 1e-10 * np.array([.06, -51., 6., 5.])

        riem_exp_2 = hyperbolic_space.riemannian_exp(ref_point_2, vector_2)
        result_2 = hyperbolic_space.riemannian_log(ref_point_2, riem_exp_2)
        expected_2 = hyperbolic_space.projection_to_tangent_space(ref_point_2,
                                                                  vector_2)

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_riemannian_dist(self):
        # Distance between a point and itself is 0.
        point_a_1 = hyperbolic_space.random_uniform(2, 1)
        point_b_1 = point_a_1
        result_1 = hyperbolic_space.riemannian_dist(point_a_1, point_b_1)
        expected_1 = 0.

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_riemannian_exp_and_dist_and_projection_to_tangent_space(self):
        ref_point_1 = hyperbolic_space.random_uniform(3, 1)
        vector_1 = np.array([9., 0., -1., -2.])
        tangent_vec_1 = hyperbolic_space.projection_to_tangent_space(
                                                           ref_point_1,
                                                           vector_1)
        riem_exp_1 = hyperbolic_space.riemannian_exp(ref_point_1,
                                                     tangent_vec_1)

        result_1 = hyperbolic_space.riemannian_dist(ref_point_1, riem_exp_1)
        expected_1 = math.sqrt(hyperbolic_space.embedding_squared_norm(
                            tangent_vec_1))

        self.assertTrue(np.allclose(result_1, expected_1))


if __name__ == '__main__':
        unittest.main()
