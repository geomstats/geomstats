"""Unit tests for hyperbolic_space module."""

from geomstats.hyperbolic_space import HyperbolicSpace

import geomstats.backend as gs
import math
import tests.helper as helper
import unittest

# Tolerance for errors on predicted vectors, relative to the *norm*
# of the vector, as opposed to the standard behavior of gs.allclose
# where it is relative to each element of the array
RTOL = 1e-6


class TestHyperbolicSpaceMethods(helper.TestGeomstatsMethods):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)

        self.dimension = 6
        self.space = HyperbolicSpace(dimension=self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10
        self.depth = 3

    def test_belongs(self):
        self.check_shape_belongs(self.space)

    def test_belongs_vectorization(self):
        self.check_shape_belongs_vectorization(
            self.space, self.n_samples)

    def test_belongs_vectorization_with_depth(self):
        self.check_shape_belongs_vectorization_with_depth(
            self.space, self.n_samples, self.depth)

    def test_random_uniform(self):
        self.check_shape_random_uniform(
            self.space, self.dimension + 1)

    def test_random_uniform_vectorization(self):
        self.check_shape_random_uniform_vectorization(
            self.space, self.n_samples, self.dimension + 1)

    def test_random_uniform_vectorization_with_depth(self):
        self.check_shape_random_uniform_vectorization_with_depth(
            self.space, self.n_samples, self.depth, self.dimension + 1)

    def test_random_uniform_and_belongs(self):
        """
        Test that the random uniform method samples
        on the hyperbolic space.
        """
        self.assert_random_uniform_and_belongs(self.space)

    def test_random_uniform_and_belongs_vectorization(self):
        """
        Test that the random uniform method samples
        on the hyperbolic space.
        """
        self.assert_random_uniform_and_belongs_vectorization(
            self.space, self.n_samples)

    def test_random_uniform_and_belongs_vectorization_with_depth(self):
        """
        Test that the random uniform method samples
        on the hyperbolic space.
        """
        self.assert_random_uniform_and_belongs_vectorization_with_depth(
            self.space, self.n_samples, self.depth)

    def test_intrinsic_and_extrinsic_coords(self):
        """
        Test that the composition of
        intrinsic_to_extrinsic_coords and
        extrinsic_to_intrinsic_coords
        gives the identity.
        """
        point_int = gs.ones(self.dimension)
        point_ext = self.space.intrinsic_to_extrinsic_coords(point_int)
        result = self.space.extrinsic_to_intrinsic_coords(point_ext)
        expected = point_int
        expected = helper.to_vector(expected)

        self.assertAllClose(result, expected)

        point_ext = self.space.random_uniform()
        point_int = self.space.extrinsic_to_intrinsic_coords(point_ext)
        result = self.space.intrinsic_to_extrinsic_coords(point_int)
        expected = point_ext
        expected = helper.to_vector(expected)

        self.assertAllClose(result, expected)

    def test_intrinsic_and_extrinsic_coords_vectorization(self):
        """
        Test that the composition of
        intrinsic_to_extrinsic_coords and
        extrinsic_to_intrinsic_coords
        gives the identity.
        """
        point_int = gs.array([[.1, 0., 0., .1, 0., 0.],
                              [.1, .1, .1, .4, .1, 0.],
                              [.1, .3, 0., .1, 0., 0.],
                              [-0.1, .1, -.4, .1, -.01, 0.],
                              [0., 0., .1, .1, -0.08, -0.1],
                              [.1, .1, .1, .1, 0., -0.5]])
        point_ext = self.space.intrinsic_to_extrinsic_coords(point_int)
        result = self.space.extrinsic_to_intrinsic_coords(point_ext)
        expected = point_int
        expected = helper.to_vector(expected)

        self.assertAllClose(result, expected)

        n_samples = self.n_samples
        point_ext = self.space.random_uniform(n_samples=n_samples)
        point_int = self.space.extrinsic_to_intrinsic_coords(point_ext)
        result = self.space.intrinsic_to_extrinsic_coords(point_int)
        expected = point_ext
        expected = helper.to_vector(expected)

        self.assertAllClose(result, expected)

    def test_log_and_exp_general_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        # General case
        base_point = self.space.random_uniform()
        point = self.space.random_uniform()

        log = self.metric.log(point=point, base_point=base_point)
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = point

        self.assertAllClose(result, expected)

    def test_exp_and_belongs(self):
        H2 = HyperbolicSpace(dimension=2)
        METRIC = H2.metric

        base_point = gs.array([1., 0., 0.])
        assert H2.belongs(base_point)

        tangent_vec = H2.projection_to_tangent_space(
                vector=gs.array([10., 200., 1.]),
                base_point=base_point)
        exp = METRIC.exp(tangent_vec=tangent_vec,
                         base_point=base_point)
        self.assertTrue(H2.belongs(exp))

    def test_exp_vectorization(self):
        self.check_shape_exp_vectorization(
            self.space, self.n_samples, self.dimension + 1)

    def test_exp_vectorization_with_depth(self):
        self.check_shape_exp_vectorization_with_depth(
            self.space, self.n_samples, self.depth, self.dimension + 1)

    def test_log_vectorization(self):
        self.check_shape_log_vectorization(
            self.space, self.n_samples, self.dimension + 1)

    def test_log_vectorization_with_depth(self):
        self.check_shape_log_vectorization_with_depth(
            self.space, self.n_samples, self.depth, self.dimension + 1)

    def test_squared_norm_and_squared_dist(self):
        """
        Test that the squared distance between two points is
        the squared norm of their logarithm.
        """
        point_a = self.space.random_uniform()
        point_b = self.space.random_uniform()
        log = self.metric.log(point=point_a, base_point=point_b)
        result = self.metric.squared_norm(vector=log)
        expected = self.metric.squared_dist(point_a, point_b)

        self.assertAllClose(result, expected)

    def test_squared_dist_vectorization(self):
        self.check_shape_squared_dist_vectorization(
            self.space, self.metric, self.n_samples)

    def test_squared_dist_vectorization_with_depth(self):
        self.check_shape_squared_dist_vectorization_with_depth(
            self.space, self.metric, self.n_samples, self.depth)

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

        self.assertAllClose(result, expected)

    def test_dist_vectorization(self):
        self.check_shape_dist_vectorization(
            self.space, self.metric, self.n_samples)

    def test_dist_vectorization_with_depth(self):
        self.check_shape_dist_vectorization_with_depth(
            self.space, self.metric, self.n_samples, self.depth)

    def test_log_and_exp_edge_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        # Edge case: two very close points, base_point_2 and point_2,
        # form an angle < epsilon
        base_point_intrinsic = gs.array([1., 2., 3., 4., 5., 6.])
        base_point = self.space.intrinsic_to_extrinsic_coords(
                                                       base_point_intrinsic)
        point_intrinsic = (base_point_intrinsic
                           + 1e-12 * gs.array([-1., -2., 1., 1., 2., 1.]))
        point = self.space.intrinsic_to_extrinsic_coords(
                                                       point_intrinsic)

        log = self.metric.log(point=point, base_point=base_point)
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = point

        self.assertAllClose(result, expected)

    def test_exp_and_log_and_projection_to_tangent_space_general_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Exp then Riemannian Log
        # General case
        base_point = self.space.random_uniform()
        # TODO(nina): this fails for high euclidean norms of vector_1
        vector = gs.array([9., 4., 0., 0., -1., -3., 2.])
        vector = self.space.projection_to_tangent_space(
                                                  vector=vector,
                                                  base_point=base_point)
        exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
        result = self.metric.log(point=exp, base_point=base_point)

        expected = vector
        norm = gs.linalg.norm(expected)
        atol = RTOL
        if norm != 0:
            atol = RTOL * norm
        self.assertTrue(gs.allclose(result, expected, atol=atol))

    def test_exp_and_log_and_projection_to_tangent_space_edge_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Exp then Riemannian Log
        # Edge case: tangent vector has norm < epsilon
        base_point = self.space.random_uniform()
        vector = 1e-10 * gs.array([.06, -51., 6., 5., 6., 6., 6.])

        exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
        result = self.metric.log(point=exp, base_point=base_point)
        expected = self.space.projection_to_tangent_space(
                                                   vector=vector,
                                                   base_point=base_point)

        self.assertAllClose(result, expected)

    def test_dist(self):
        # Distance between a point and itself is 0.
        point_a = self.space.random_uniform()
        point_b = point_a
        result = self.metric.dist(point_a, point_b)
        expected = 0.

        self.assertAllClose(result, expected)

    def test_exp_and_dist_and_projection_to_tangent_space(self):
        # TODO(nina): this fails for high norms of vector
        base_point = self.space.random_uniform()
        vector = gs.array([2., 0., -1., -2., 7., 4., 1.])
        tangent_vec = self.space.projection_to_tangent_space(
                                                vector=vector,
                                                base_point=base_point)
        exp = self.metric.exp(tangent_vec=tangent_vec,
                              base_point=base_point)

        result = self.metric.dist(base_point, exp)
        sq_norm = self.metric.embedding_metric.squared_norm(
                                                 tangent_vec)
        expected = math.sqrt(sq_norm)
        self.assertAllClose(result, expected)

    def test_geodesic_and_belongs(self):
        # TODO(nina): this tests fails when geodesic goes "too far"
        initial_point = self.space.random_uniform()
        vector = gs.array([2., 0., -1., -2., 7., 4., 1.])
        initial_tangent_vec = self.space.projection_to_tangent_space(
                                            vector=vector,
                                            base_point=initial_point)
        geodesic = self.metric.geodesic(
                                   initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)

        t = gs.linspace(start=0, stop=1, num=100)
        points = geodesic(t)
        self.assertTrue(gs.all(self.space.belongs(points)))

    def test_variance(self):
        point = self.space.random_uniform()
        result = self.metric.variance([point, point])
        expected = 0

        self.assertAllClose(result, expected)

    def test_mean(self):
        point = self.space.random_uniform()
        result = self.metric.mean([point, point])
        expected = point

        self.assertAllClose(result, expected)

    def test_mean_and_belongs(self):
        point_a = self.space.random_uniform()
        point_b = self.space.random_uniform()
        point_c = self.space.random_uniform()

        result = self.metric.mean([point_a, point_b, point_c])
        self.assertTrue(self.space.belongs(result))


if __name__ == '__main__':
        unittest.main()
