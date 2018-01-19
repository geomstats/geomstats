"""Unit tests for hyperbolic_space module."""

import math
import numpy as np
import unittest

from geomstats.hyperbolic_space import HyperbolicMetric
from geomstats.hyperbolic_space import HyperbolicSpace


class TestHyperbolicSpaceMethods(unittest.TestCase):
    DIMENSION = 6
    METRIC = HyperbolicMetric()
    SPACE = HyperbolicSpace(dimension=DIMENSION)

    def test_random_uniform_and_belongs(self):
        """
        Test that the random uniform method samples
        on the hyperbolic space.
        """
        ref_point = self.SPACE.random_uniform(self.DIMENSION, 1)
        assert self.SPACE.belongs(ref_point)

    def test_riemannian_log_and_exp(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        # General case
        ref_point_1 = self.SPACE.random_uniform(self.DIMENSION, 1)
        point_1 = self.SPACE.random_uniform(self.DIMENSION, 1)

        riem_log_1 = self.METRIC.riemannian_log(ref_point_1, point_1)
        result_1 = self.METRIC.riemannian_exp(ref_point_1, riem_log_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

        # Riemannian Log then Riemannian Exp
        # Edge case: two very close points, ref_point_2 and point_2,
        # form an angle < epsilon
        ref_point_intrinsic_2 = np.array([1., 2., 3., 4., 5., 6.])
        ref_point_2 = self.SPACE.intrinsic_to_extrinsic_coords(
                                                       ref_point_intrinsic_2)
        point_intrinsic_2 = (ref_point_intrinsic_2
                             + 1e-12 * np.array([-1., -2., 1., 1., 2., 1.]))
        point_2 = self.SPACE.intrinsic_to_extrinsic_coords(
                                                       point_intrinsic_2)

        riem_log_2 = self.METRIC.riemannian_log(ref_point_2, point_2)
        result_2 = self.METRIC.riemannian_exp(ref_point_2, riem_log_2)
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
        ref_point_1 = self.SPACE.random_uniform(self.DIMENSION, 1)
        # TODO(nina): this fails for high euclidean norms of vector_1
        vector_1 = np.array([9., 4., 0., 0., -1., -3., 2.])
        vector_1 = self.SPACE.projection_to_tangent_space(ref_point_1,
                                                          vector_1)
        riem_exp_1 = self.METRIC.riemannian_exp(ref_point_1, vector_1)
        result_1 = self.METRIC.riemannian_log(ref_point_1, riem_exp_1)

        expected_1 = vector_1
        self.assertTrue(np.allclose(result_1, expected_1))

        # Riemannian Exp then Riemannian Log
        # Edge case: tangent vector has norm < epsilon
        ref_point_2 = self.SPACE.random_uniform(self.DIMENSION, 1)
        vector_2 = 1e-10 * np.array([.06, -51., 6., 5., 6., 6., 6.])

        riem_exp_2 = self.METRIC.riemannian_exp(ref_point_2, vector_2)
        result_2 = self.METRIC.riemannian_log(ref_point_2, riem_exp_2)
        expected_2 = self.SPACE.projection_to_tangent_space(ref_point_2,
                                                            vector_2)

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_riemannian_dist(self):
        # Distance between a point and itself is 0.
        point_a_1 = self.SPACE.random_uniform(self.DIMENSION, 1)
        point_b_1 = point_a_1
        result_1 = self.METRIC.riemannian_dist(point_a_1, point_b_1)
        expected_1 = 0.

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_riemannian_exp_and_dist_and_projection_to_tangent_space(self):
        # TODO(nina): this fails for high norms of vector_1
        ref_point_1 = self.SPACE.random_uniform(self.DIMENSION, 1)
        vector_1 = np.array([2., 0., -1., -2., 7., 4., 1.])
        tangent_vec_1 = self.SPACE.projection_to_tangent_space(
                                                           ref_point_1,
                                                           vector_1)
        riem_exp_1 = self.METRIC.riemannian_exp(ref_point_1,
                                                tangent_vec_1)

        result_1 = self.METRIC.riemannian_dist(ref_point_1, riem_exp_1)
        sq_norm = self.METRIC.embedding_metric.riemannian_squared_norm(
                                                 tangent_vec_1)
        expected_1 = math.sqrt(sq_norm)
        self.assertTrue(np.allclose(result_1, expected_1))


if __name__ == '__main__':
        unittest.main()
