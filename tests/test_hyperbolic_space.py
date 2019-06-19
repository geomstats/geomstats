"""
Unit tests for the Hyperbolic space.
"""

import math
import numpy as np

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper

from geomstats.hyperbolic_space import HyperbolicSpace
from geomstats.minkowski_space import MinkowskiSpace

# Tolerance for errors on predicted vectors, relative to the *norm*
# of the vector, as opposed to the standard behavior of gs.allclose
# where it is relative to each element of the array
RTOL = 1e-6


class TestHyperbolicSpaceMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        self.dimension = 3
        self.space = HyperbolicSpace(dimension=self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10

    def test_random_uniform_and_belongs(self):
        point = self.space.random_uniform()
        result = self.space.belongs(point)
        expected = gs.array([[True]])

        self.assertAllClose(result, expected)

    def test_random_uniform(self):
        result = self.space.random_uniform()

        self.assertAllClose(gs.shape(result), (1, self.dimension + 1))

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

        point_ext = gs.array([2.0, 1.0, 1.0, 1.0])
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

        point_ext = gs.array([[2., 1., 1., 1.],
                              [4., 1., 3., math.sqrt(5.)],
                              [3., 2., 0., 2.]])
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
        base_point = gs.array([4.0, 1., 3.0, math.sqrt(5.)])
        point = gs.array([2.0, 1.0, 1.0, 1.0])

        log = self.metric.log(point=point, base_point=base_point)

        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = helper.to_vector(point)
        self.assertAllClose(result, expected)

    def test_exp_and_belongs(self):
        H2 = HyperbolicSpace(dimension=2)
        METRIC = H2.metric

        base_point = gs.array([1., 0., 0.])
        with self.session():
            self.assertTrue(gs.eval(H2.belongs(base_point)))

        tangent_vec = H2.projection_to_tangent_space(
                vector=gs.array([1., 2., 1.]),
                base_point=base_point)
        exp = METRIC.exp(tangent_vec=tangent_vec,
                         base_point=base_point)
        with self.session():
            self.assertTrue(gs.eval(H2.belongs(exp)))

    def test_exp_vectorization(self):
        n_samples = 3
        dim = self.dimension + 1

        one_vec = gs.array([2.0, 1.0, 1.0, 1.0])
        one_base_point = gs.array([4.0, 3., 1.0, math.sqrt(5)])
        n_vecs = gs.array([[2., 1., 1., 1.],
                           [4., 1., 3., math.sqrt(5.)],
                           [3., 2., 0., 2.]])
        n_base_points = gs.array([
            [2.0, 0.0, 1.0, math.sqrt(2)],
            [5.0, math.sqrt(8), math.sqrt(8), math.sqrt(8)],
            [1.0, 0.0, 0.0, 0.0]])

        one_tangent_vec = self.space.projection_to_tangent_space(
            one_vec, base_point=one_base_point)
        result = self.metric.exp(one_tangent_vec, one_base_point)
        self.assertAllClose(gs.shape(result), (1, dim))

        n_tangent_vecs = self.space.projection_to_tangent_space(
            n_vecs, base_point=one_base_point)
        result = self.metric.exp(n_tangent_vecs, one_base_point)
        self.assertAllClose(gs.shape(result),  (n_samples, dim))

        expected = np.zeros((n_samples, dim))

        with self.session():
            for i in range(n_samples):
                expected[i] = gs.eval(
                    self.metric.exp(n_tangent_vecs[i], one_base_point))
            expected = helper.to_vector(gs.array(expected))
            self.assertAllClose(result, expected)

        one_tangent_vec = self.space.projection_to_tangent_space(
            one_vec, base_point=n_base_points)
        result = self.metric.exp(one_tangent_vec, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        expected = np.zeros((n_samples, dim))
        with self.session():
            for i in range(n_samples):
                expected[i] = gs.eval(self.metric.exp(one_tangent_vec[i],
                                      n_base_points[i]))
            expected = helper.to_vector(gs.array(expected))
            self.assertAllClose(result, expected)

        n_tangent_vecs = self.space.projection_to_tangent_space(
            n_vecs, base_point=n_base_points)
        result = self.metric.exp(n_tangent_vecs, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        expected = np.zeros((n_samples, dim))
        with self.session():
            for i in range(n_samples):
                expected[i] = gs.eval(self.metric.exp(n_tangent_vecs[i],
                                      n_base_points[i]))
            expected = helper.to_vector(gs.array(expected))
            self.assertAllClose(result, expected)

    def test_log_vectorization(self):
        n_samples = 3
        dim = self.dimension + 1

        one_point = gs.array([2.0, 1.0, 1.0, 1.0])
        one_base_point = gs.array([4.0, 3., 1.0, math.sqrt(5)])
        n_points = gs.array([[2.0, 1.0, 1.0, 1.0],
                             [4.0, 1., 3.0, math.sqrt(5)],
                             [3.0, 2.0, 0.0, 2.0]])
        n_base_points = gs.array([
            [2.0, 0.0, 1.0, math.sqrt(2)],
            [5.0, math.sqrt(8), math.sqrt(8), math.sqrt(8)],
            [1.0, 0.0, 0.0, 0.0]])

        result = self.metric.log(one_point, one_base_point)
        self.assertAllClose(gs.shape(result), (1, dim))

        result = self.metric.log(n_points, one_base_point)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.log(one_point, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.log(n_points, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

    def test_inner_product(self):
        """
        Test that the inner product between two tangent vectors
        is the Minkowski inner product.
        """
        minkowski_space = MinkowskiSpace(self.dimension+1)
        base_point = gs.array(
                [1.16563816,  0.36381045, -0.47000603,  0.07381469])

        tangent_vec_a = self.space.projection_to_tangent_space(
                vector=gs.array([10., 200., 1., 1.]),
                base_point=base_point)

        tangent_vec_b = self.space.projection_to_tangent_space(
                vector=gs.array([11., 20., -21., 0.]),
                base_point=base_point)

        result = self.metric.inner_product(
                tangent_vec_a, tangent_vec_b, base_point)

        expected = minkowski_space.metric.inner_product(
                tangent_vec_a, tangent_vec_b, base_point)

        with self.session():
            self.assertAllClose(result, expected)

    def test_squared_norm_and_squared_dist(self):
        """
        Test that the squared distance between two points is
        the squared norm of their logarithm.
        """
        point_a = gs.array([2.0, 1.0, 1.0, 1.0])
        point_b = gs.array([4.0, 1., 3.0, math.sqrt(5)])
        log = self.metric.log(point=point_a, base_point=point_b)
        result = self.metric.squared_norm(vector=log)
        expected = self.metric.squared_dist(point_a, point_b)

        with self.session():
            self.assertAllClose(result, expected)

    def test_norm_and_dist(self):
        """
        Test that the distance between two points is
        the norm of their logarithm.
        """
        point_a = gs.array([2.0, 1.0, 1.0, 1.0])
        point_b = gs.array([4.0, 1., 3.0, math.sqrt(5)])
        log = self.metric.log(point=point_a, base_point=point_b)
        result = self.metric.norm(vector=log)
        expected = self.metric.dist(point_a, point_b)

        with self.session():
            self.assertAllClose(result, expected)

    def test_log_and_exp_edge_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        # Edge case: two very close points, base_point_2 and point_2,
        # form an angle < epsilon
        base_point_intrinsic = gs.array([1., 2., 3.])
        base_point = self.space.intrinsic_to_extrinsic_coords(
                                                       base_point_intrinsic)
        point_intrinsic = (base_point_intrinsic
                           + 1e-12 * gs.array([-1., -2., 1.]))
        point = self.space.intrinsic_to_extrinsic_coords(
                                                       point_intrinsic)

        log = self.metric.log(point=point, base_point=base_point)
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = point

        with self.session():
            self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_exp_and_log_and_projection_to_tangent_space_general_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Exp then Riemannian Log
        # General case
        base_point = gs.array([4.0, 1., 3.0, math.sqrt(5)])
        vector = gs.array([2.0, 1.0, 1.0, 1.0])
        vector = self.space.projection_to_tangent_space(
                                                  vector=vector,
                                                  base_point=base_point)
        exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
        result = self.metric.log(point=exp, base_point=base_point)

        expected = vector
        with self.session():
            self.assertAllClose(result, expected)

    def test_dist(self):
        # Distance between a point and itself is 0.
        point_a = gs.array([4.0, 1., 3.0, math.sqrt(5)])
        point_b = point_a
        result = self.metric.dist(point_a, point_b)
        expected = gs.array([[0]])

        with self.session():
            self.assertAllClose(result, expected)

    def test_exp_and_dist_and_projection_to_tangent_space(self):
        base_point = gs.array([4.0, 1., 3.0, math.sqrt(5)])
        vector = gs.array([0.001, 0., -.00001, -.00003])
        tangent_vec = self.space.projection_to_tangent_space(
                                                vector=vector,
                                                base_point=base_point)
        exp = self.metric.exp(tangent_vec=tangent_vec,
                              base_point=base_point)

        result = self.metric.dist(base_point, exp)
        sq_norm = self.metric.embedding_metric.squared_norm(
                                                 tangent_vec)
        expected = sq_norm
        with self.session():
            self.assertAllClose(result, expected, atol=1e-2)

    def test_geodesic_and_belongs(self):
        # TODO(nina): Fix this tests, as it fails when geodesic goes "too far"
        initial_point = gs.array([4.0, 1., 3.0, math.sqrt(5)])
        n_geodesic_points = 100
        vector = gs.array([1., 0., 0., 0.])

        initial_tangent_vec = self.space.projection_to_tangent_space(
                                            vector=vector,
                                            base_point=initial_point)
        geodesic = self.metric.geodesic(
                                   initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)

        t = gs.linspace(start=0., stop=1., num=n_geodesic_points)
        points = geodesic(t)

        result = self.space.belongs(points)
        expected = gs.array(n_geodesic_points * [[True]])

        with self.session():
            self.assertAllClose(expected, result)

    def test_exp_and_log_and_projection_to_tangent_space_edge_case(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Exp then Riemannian Log
        # Edge case: tangent vector has norm < epsilon
        base_point = gs.array([2., 1., 1., 1.])
        vector = 1e-10 * gs.array([.06, -51., 6., 5.])

        exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
        result = self.metric.log(point=exp, base_point=base_point)
        expected = self.space.projection_to_tangent_space(
                                                   vector=vector,
                                                   base_point=base_point)

        self.assertAllClose(result, expected, atol=1e-8)

    @geomstats.tests.np_and_tf_only
    def test_variance(self):
        point = gs.array([2., 1., 1., 1.])
        points = gs.array([point, point])
        result = self.metric.variance(points)
        expected = helper.to_scalar(0.)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_mean(self):
        point = gs.array([2., 1., 1., 1.])
        points = gs.array([point, point])
        result = self.metric.mean(points)
        expected = helper.to_vector(point)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_mean_and_belongs(self):
        point_a = self.space.random_uniform()
        point_b = self.space.random_uniform()
        point_c = self.space.random_uniform()
        points = gs.concatenate([point_a, point_b, point_c], axis=0)

        mean = self.metric.mean(points)
        result = self.space.belongs(mean)
        expected = gs.array([[True]])

        self.assertAllClose(result, expected)


if __name__ == '__main__':
    geomstats.tests.main()
