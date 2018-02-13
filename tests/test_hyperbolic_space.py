"""Unit tests for hyperbolic_space module."""

import math
import numpy as np
import unittest

from geomstats.hyperbolic_space import HyperbolicSpace

# Tolerance for errors on predicted vectors, relative to the *norm*
# of the vector, as opposed to the standard behavior of np.allclose
# where it is relative to each element of the array
RTOL = 1e-6


class TestHyperbolicSpaceMethods(unittest.TestCase):
    def setUp(self):
        self.dimension = 6
        self.space = HyperbolicSpace(dimension=self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10

    def test_random_uniform_and_belongs(self):
        """
        Test that the random uniform method samples
        on the hyperbolic space.
        """
        point = self.space.random_uniform()
        self.assertTrue(self.space.belongs(point))

    def test_random_uniform_and_belongs_vectorization(self):
        """
        Test that the random uniform method samples
        on the hyperbolic space.
        """
        n_samples = self.n_samples
        points = self.space.random_uniform(n_samples=n_samples)
        self.assertTrue(np.all(self.space.belongs(points)))

    def test_intrinsic_and_extrinsic_coords(self):
        """
        Test that the composition of
        intrinsic_to_extrinsic_coords and
        extrinsic_to_intrinsic_coords
        gives the identity.
        """
        point_int = np.ones(self.dimension)
        point_ext = self.space.intrinsic_to_extrinsic_coords(point_int)
        result = self.space.extrinsic_to_intrinsic_coords(point_ext)
        expected = point_int

        self.assertTrue(np.allclose(result, expected))

        point_ext = self.space.random_uniform()
        point_int = self.space.extrinsic_to_intrinsic_coords(point_ext)
        result = self.space.intrinsic_to_extrinsic_coords(point_int)
        expected = point_ext

        self.assertTrue(np.allclose(result, expected))

        self.assertTrue(np.allclose(result, expected))

    def test_intrinsic_and_extrinsic_coords_vectorization(self):
        """
        Test that the composition of
        intrinsic_to_extrinsic_coords and
        extrinsic_to_intrinsic_coords
        gives the identity.
        """
        point_int = np.array([[.1, 0., 0., .1, 0., 0.],
                              [.1, .1, .1, .4, .1, 0.],
                              [.1, .3, 0., .1, 0., 0.],
                              [-0.1, .1, -.4, .1, -.01, 0.],
                              [0., 0., .1, .1, -0.08, -0.1],
                              [.1, .1, .1, .1, 0., -0.5]])
        point_ext = self.space.intrinsic_to_extrinsic_coords(point_int)
        result = self.space.extrinsic_to_intrinsic_coords(point_ext)
        expected = point_int

        self.assertTrue(np.allclose(result, expected))

        n_samples = self.n_samples
        point_ext = self.space.random_uniform(n_samples=n_samples)
        point_int = self.space.extrinsic_to_intrinsic_coords(point_ext)
        result = self.space.intrinsic_to_extrinsic_coords(point_int)
        expected = point_ext

        self.assertTrue(np.allclose(result, expected))

    def test_log_and_exp_general_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        # General case
        base_point_1 = self.space.random_uniform()
        point_1 = self.space.random_uniform()

        log_1 = self.metric.log(point=point_1, base_point=base_point_1)
        result_1 = self.metric.exp(tangent_vec=log_1, base_point=base_point_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_exp_vectorization(self):
        n_samples = self.n_samples
        dim = self.dimension
        one_tangent_vec = self.space.random_uniform(n_samples=1)
        one_base_point = self.space.random_uniform(n_samples=1)
        n_tangent_vecs = self.space.random_uniform(n_samples=n_samples)
        n_base_points = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.exp(one_tangent_vec, one_base_point)
        self.assertTrue(np.allclose(result.shape, (1, dim + 1)))

        result = self.metric.exp(n_tangent_vecs, one_base_point)
        self.assertTrue(np.allclose(result.shape, (n_samples, dim + 1)),
                        '\n result.shape = {}'.format(result.shape))

        result = self.metric.exp(one_tangent_vec, n_base_points)
        self.assertTrue(np.allclose(result.shape, (n_samples, dim + 1)))

        result = self.metric.exp(n_tangent_vecs, n_base_points)
        self.assertTrue(np.allclose(result.shape, (n_samples, dim + 1)))

    def test_log_vectorization(self):
        n_samples = self.n_samples
        dim = self.dimension
        one_point = self.space.random_uniform(n_samples=1)
        one_base_point = self.space.random_uniform(n_samples=1)
        n_points = self.space.random_uniform(n_samples=n_samples)
        n_base_points = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.log(one_point, one_base_point)
        self.assertTrue(np.allclose(result.shape, (1, dim + 1)))

        result = self.metric.log(n_points, one_base_point)
        self.assertTrue(np.allclose(result.shape, (n_samples, dim + 1)))

        result = self.metric.log(one_point, n_base_points)
        self.assertTrue(np.allclose(result.shape, (n_samples, dim + 1)))

        result = self.metric.log(n_points, n_base_points)
        self.assertTrue(np.allclose(result.shape, (n_samples, dim + 1)))

    def test_squared_norm_and_squared_dist(self):
        """
        Test that the squqred distance between two points is
        the squared norm of their logarithm.
        """
        point_a = self.space.random_uniform()
        point_b = self.space.random_uniform()
        log = self.metric.log(point=point_a, base_point=point_b)
        result = self.metric.squared_norm(vector=log)
        expected = self.metric.squared_dist(point_a, point_b)

        self.assertTrue(np.allclose(result, expected))

    def test_squared_dist_vectorization(self):
        n_samples = self.n_samples
        one_point_a = self.space.random_uniform(n_samples=1)
        one_point_b = self.space.random_uniform(n_samples=1)
        n_points_a = self.space.random_uniform(n_samples=n_samples)
        n_points_b = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.squared_dist(one_point_a, one_point_b)
        self.assertTrue(np.allclose(result.shape, (1, 1)))

        result = self.metric.squared_dist(n_points_a, one_point_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

        result = self.metric.squared_dist(one_point_a, n_points_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

        result = self.metric.squared_dist(n_points_a, n_points_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

    def test_norm_and_dist(self):
        """
        Test that the distance between two points is
        the norm of their logarithm.
        """
        point_a = self.space.random_uniform()
        point_b = self.space.random_uniform()
        log = self.metric.log(point=point_a, base_point=point_b)
        result = self.metric.norm(vector=log)
        expected = self.metric.dist(point_a, point_b)

        self.assertTrue(np.allclose(result, expected))

    def test_dist_vectorization(self):
        n_samples = self.n_samples
        one_point_a = self.space.random_uniform(n_samples=1)
        one_point_b = self.space.random_uniform(n_samples=1)
        n_points_a = self.space.random_uniform(n_samples=n_samples)
        n_points_b = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.dist(one_point_a, one_point_b)
        self.assertTrue(np.allclose(result.shape, (1, 1)))

        result = self.metric.dist(n_points_a, one_point_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

        result = self.metric.dist(one_point_a, n_points_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

        result = self.metric.dist(n_points_a, n_points_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

    def test_log_and_exp_edge_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        # Edge case: two very close points, base_point_2 and point_2,
        # form an angle < epsilon
        base_point_intrinsic_2 = np.array([1., 2., 3., 4., 5., 6.])
        base_point_2 = self.space.intrinsic_to_extrinsic_coords(
                                                       base_point_intrinsic_2)
        point_intrinsic_2 = (base_point_intrinsic_2
                             + 1e-12 * np.array([-1., -2., 1., 1., 2., 1.]))
        point_2 = self.space.intrinsic_to_extrinsic_coords(
                                                       point_intrinsic_2)

        log_2 = self.metric.log(point=point_2, base_point=base_point_2)
        result_2 = self.metric.exp(tangent_vec=log_2, base_point=base_point_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_exp_and_log_and_projection_to_tangent_space_general_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Exp then Riemannian Log
        # General case
        base_point_1 = self.space.random_uniform()
        # TODO(nina): this fails for high euclidean norms of vector_1
        vector_1 = np.array([9., 4., 0., 0., -1., -3., 2.])
        vector_1 = self.space.projection_to_tangent_space(
                                                  vector=vector_1,
                                                  base_point=base_point_1)
        exp_1 = self.metric.exp(tangent_vec=vector_1, base_point=base_point_1)
        result_1 = self.metric.log(point=exp_1, base_point=base_point_1)

        expected_1 = vector_1
        norm = np.linalg.norm(expected_1)
        atol = RTOL
        if norm != 0:
            atol = RTOL * norm
        self.assertTrue(np.allclose(result_1, expected_1, atol=atol))

    def test_exp_and_log_and_projection_to_tangent_space_edge_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Exp then Riemannian Log
        # Edge case: tangent vector has norm < epsilon
        base_point_2 = self.space.random_uniform()
        vector_2 = 1e-10 * np.array([.06, -51., 6., 5., 6., 6., 6.])

        exp_2 = self.metric.exp(tangent_vec=vector_2, base_point=base_point_2)
        result_2 = self.metric.log(point=exp_2, base_point=base_point_2)
        expected_2 = self.space.projection_to_tangent_space(
                                                   vector=vector_2,
                                                   base_point=base_point_2)

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_dist(self):
        # Distance between a point and itself is 0.
        point_a_1 = self.space.random_uniform()
        point_b_1 = point_a_1
        result_1 = self.metric.dist(point_a_1, point_b_1)
        expected_1 = 0.

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_exp_and_dist_and_projection_to_tangent_space(self):
        # TODO(nina): this fails for high norms of vector_1
        base_point_1 = self.space.random_uniform()
        vector_1 = np.array([2., 0., -1., -2., 7., 4., 1.])
        tangent_vec_1 = self.space.projection_to_tangent_space(
                                                vector=vector_1,
                                                base_point=base_point_1)
        exp_1 = self.metric.exp(tangent_vec=tangent_vec_1,
                                base_point=base_point_1)

        result_1 = self.metric.dist(base_point_1, exp_1)
        sq_norm = self.metric.embedding_metric.squared_norm(
                                                 tangent_vec_1)
        expected_1 = math.sqrt(sq_norm)
        self.assertTrue(np.allclose(result_1, expected_1))

    def test_variance(self):
        point = self.space.random_uniform()
        result = self.metric.variance([point, point])
        expected = 0

        self.assertTrue(np.allclose(result, expected))

    def test_mean(self):
        point = self.space.random_uniform()
        result = self.metric.mean([point, point])
        expected = point

        self.assertTrue(np.allclose(result, expected))

    def test_mean_and_belongs(self):
        point_a = self.space.random_uniform()
        point_b = self.space.random_uniform()
        point_c = self.space.random_uniform()

        result = self.metric.mean([point_a, point_b, point_c])
        self.assertTrue(self.space.belongs(result))


if __name__ == '__main__':
        unittest.main()
