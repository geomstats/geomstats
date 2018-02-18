"""Unit tests for hypersphere module."""

from geomstats.hypersphere import Hypersphere

import numpy as np
import unittest


class TestHypersphereMethods(unittest.TestCase):
    def setUp(self):
        self.dimension = 4
        self.space = Hypersphere(dimension=self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10

    def test_random_uniform_and_belongs(self):
        """
        Test that the random uniform method samples
        on the hypersphere space.
        """
        point = self.space.random_uniform()
        self.assertTrue(self.space.belongs(point))

    def test_random_uniform_and_belongs_vectorization(self):
        """
        Test that the random uniform method samples
        on the hypersphere space.
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
        point_int = np.array([.1, 0., 0., .1])
        point_ext = self.space.intrinsic_to_extrinsic_coords(point_int)
        result = self.space.extrinsic_to_intrinsic_coords(point_ext)
        expected = point_int

        self.assertTrue(np.allclose(result, expected))

        point_ext = self.space.random_uniform()
        point_int = self.space.extrinsic_to_intrinsic_coords(point_ext)
        result = self.space.intrinsic_to_extrinsic_coords(point_int)
        expected = point_ext

        self.assertTrue(np.allclose(result, expected))

    def test_intrinsic_and_extrinsic_coords_vectorization(self):
        """
        Test that the composition of
        intrinsic_to_extrinsic_coords and
        extrinsic_to_intrinsic_coords
        gives the identity.
        """
        point_int = np.array([[.1, 0., 0., .1],
                              [.1, .1, .1, .4],
                              [.1, .3, 0., .1],
                              [-0.1, .1, -.4, .1],
                              [0., 0., .1, .1],
                              [.1, .1, .1, .1]])
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

        NB: points on the n-dimensional sphere are
        (n+1)-D vectors of norm 1.
        """
        # Riemannian Log then Riemannian Exp
        # General case
        base_point_1 = np.array([1., 2., 3., 4., 6.])
        base_point_1 = base_point_1 / np.linalg.norm(base_point_1)
        point_1 = np.array([0., 5., 6., 2., -1])
        point_1 = point_1 / np.linalg.norm(point_1)

        log_1 = self.metric.log(point=point_1, base_point=base_point_1)
        result_1 = self.metric.exp(tangent_vec=log_1, base_point=base_point_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_log_and_exp_edge_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.

        NB: points on the n-dimensional sphere are
        (n+1)-D vectors of norm 1.
        """
        # Riemannian Log then Riemannian Exp
        # Edge case: two very close points, base_point_2 and point_2,
        # form an angle < epsilon
        base_point_2 = np.array([1., 2., 3., 4., 6.])
        base_point_2 = base_point_2 / np.linalg.norm(base_point_2)
        point_2 = base_point_2 + 1e-12 * np.array([-1., -2., 1., 1., .1])
        point_2 = point_2 / np.linalg.norm(point_2)

        log_2 = self.metric.log(point=point_2, base_point=base_point_2)
        result_2 = self.metric.exp(tangent_vec=log_2, base_point=base_point_2)
        expected_2 = point_2

        self.assertTrue(np.allclose(result_2, expected_2))

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

    def test_exp_and_log_and_projection_to_tangent_space_general_case(self):
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
        base_point_1 = np.array([0., -3., 0., 3., 4.])
        base_point_1 = base_point_1 / np.linalg.norm(base_point_1)
        vector_1 = np.array([9., 5., 0., 0., -1.])
        vector_1 = self.space.projection_to_tangent_space(
                                                   vector=vector_1,
                                                   base_point=base_point_1)

        exp_1 = self.metric.exp(tangent_vec=vector_1, base_point=base_point_1)
        result_1 = self.metric.log(point=exp_1, base_point=base_point_1)

        expected_1 = vector_1
        norm_expected_1 = np.linalg.norm(expected_1)
        regularized_norm_expected_1 = np.mod(norm_expected_1, 2 * np.pi)
        expected_1 = expected_1 / norm_expected_1 * regularized_norm_expected_1
        # TODO(nina): this test fails
        # self.assertTrue(np.allclose(result_1, expected_1))

    def test_exp_and_log_and_projection_to_tangent_space_edge_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.

        NB: points on the n-dimensional sphere are
        (n+1)-D vectors of norm 1.
        """
        # Riemannian Exp then Riemannian Log
        # Edge case: tangent vector has norm < epsilon
        base_point_2 = np.array([10., -2., -.5, 34., 3.])
        base_point_2 = base_point_2 / np.linalg.norm(base_point_2)
        vector_2 = 1e-10 * np.array([.06, -51., 6., 5., 3.])
        vector_2 = self.space.projection_to_tangent_space(
                                                    vector=vector_2,
                                                    base_point=base_point_2)

        exp_2 = self.metric.exp(tangent_vec=vector_2, base_point=base_point_2)
        result_2 = self.metric.log(point=exp_2, base_point=base_point_2)
        expected_2 = self.space.projection_to_tangent_space(
                                                    vector=vector_2,
                                                    base_point=base_point_2)

        self.assertTrue(np.allclose(result_2, expected_2))

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

    def test_dist_point_and_itself(self):
        # Distance between a point and itself is 0.
        point_a_1 = np.array([10., -2., -.5, 2., 3.])
        point_b_1 = point_a_1
        result_1 = self.metric.dist(point_a_1, point_b_1)
        expected_1 = 0.

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_dist_orthogonal_points(self):
        # Distance between two orthogonal points is pi / 2.
        point_a_2 = np.array([10., -2., -.5, 0., 0.])
        point_b_2 = np.array([2., 10, 0., 0., 0.])
        assert np.dot(point_a_2, point_b_2) == 0

        result_2 = self.metric.dist(point_a_2, point_b_2)
        expected_2 = np.pi / 2

        self.assertTrue(np.allclose(result_2, expected_2))

    def test_exp_and_dist_and_projection_to_tangent_space(self):
        base_point_1 = np.array([16., -2., -2.5, 84., 3.])
        base_point_1 = base_point_1 / np.linalg.norm(base_point_1)

        vector_1 = np.array([9., 0., -1., -2., 1.])
        tangent_vec_1 = self.space.projection_to_tangent_space(
                                                      vector=vector_1,
                                                      base_point=base_point_1)
        exp_1 = self.metric.exp(tangent_vec=tangent_vec_1,
                                base_point=base_point_1)

        result_1 = self.metric.dist(base_point_1, exp_1)
        expected_1 = np.mod(np.linalg.norm(tangent_vec_1), 2 * np.pi)

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_geodesic_and_belongs(self):
        initial_point = self.space.random_uniform()
        vector = np.array([2., 0., -1., -2., 1.])
        initial_tangent_vec = self.space.projection_to_tangent_space(
                                            vector=vector,
                                            base_point=initial_point)
        geodesic = self.metric.geodesic(
                                   initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)

        t = np.linspace(start=0, stop=1, num=100)
        points = geodesic(t)
        self.assertTrue(np.all(self.space.belongs(points)))

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
