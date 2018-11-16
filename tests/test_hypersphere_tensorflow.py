"""
Unit tests for the hypersphere on tensorflow backend.
"""

import importlib
import numpy as np
import os
import tensorflow as tf

import geomstats.backend as gs
import tests.helper as helper

from geomstats.hypersphere import Hypersphere


class TestHypersphereTensorFlow(tf.test.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        self.dimension = 4
        self.space = Hypersphere(dimension=self.dimension)
        self.metric = self.space.metric
        self.n_samples = 3

    @classmethod
    def setUpClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        importlib.reload(gs)

    @classmethod
    def tearDownClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'numpy'
        importlib.reload(gs)

    def test_belongs(self):
        point = self.space.random_uniform()
        bool_belongs = self.space.belongs(point)
        expected = tf.convert_to_tensor([[True]])

        with self.test_session():
            self.assertAllClose(gs.eval(expected), gs.eval(bool_belongs))

    def test_random_uniform(self):
        point = self.space.random_uniform()
        point_numpy = np.random.uniform(size=(1, self.dimension + 1))

        with self.test_session():
            self.assertShapeEqual(point_numpy, point)

    def test_random_uniform_and_belongs(self):
        """
        Test that the random uniform method samples
        on the hypersphere space.
        """
        point = self.space.random_uniform()
        with self.test_session():
            self.assertTrue(gs.eval(self.space.belongs(point)[0, 0]))

    def test_projection_and_belongs(self):
        point = tf.convert_to_tensor([1., 2., 3., 4., 5.])
        result = self.space.projection(point)

        with self.test_session():
            self.assertTrue(gs.eval(self.space.belongs(result)[0, 0]))

    def test_intrinsic_and_extrinsic_coords(self):
        """
        Test that the composition of
        intrinsic_to_extrinsic_coords and
        extrinsic_to_intrinsic_coords
        gives the identity.
        """
        point_int = tf.convert_to_tensor([.1, 0., 0., .1])
        point_ext = self.space.intrinsic_to_extrinsic_coords(point_int)
        result = self.space.extrinsic_to_intrinsic_coords(point_ext)
        expected = point_int
        expected = helper.to_vector(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        # TODO(nina): Fix that the test fails if point_ext generated
        # with tf.random_uniform
        point_ext = (1. / (gs.sqrt(6.))
                     * tf.convert_to_tensor([1., 0., 0., 1., 2.]))
        point_int = self.space.extrinsic_to_intrinsic_coords(point_ext)
        result = self.space.intrinsic_to_extrinsic_coords(point_int)
        expected = point_ext
        expected = helper.to_vector(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_intrinsic_and_extrinsic_coords_vectorization(self):
        """
        Test that the composition of
        intrinsic_to_extrinsic_coords and
        extrinsic_to_intrinsic_coords
        gives the identity.
        """
        point_int = tf.convert_to_tensor(
                [[.1, 0., 0., .1],
                 [.1, .1, .1, .4],
                 [.1, .3, 0., .1],
                 [-0.1, .1, -.4, .1],
                 [0., 0., .1, .1],
                 [.1, .1, .1, .1]])
        point_ext = self.space.intrinsic_to_extrinsic_coords(point_int)
        result = self.space.extrinsic_to_intrinsic_coords(point_ext)
        expected = point_int
        expected = helper.to_vector(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        sqrt_3 = np.sqrt(3.)
        point_ext = tf.convert_to_tensor(
            [[1. / sqrt_3, 0., 0., 1. / sqrt_3, 1. / sqrt_3],
             [1. / sqrt_3, 1. / sqrt_3, 1. / sqrt_3, 0., 0.],
             [0., 0., 1. / sqrt_3, 1. / sqrt_3, 1. / sqrt_3]],
            dtype=np.float64)

        point_int = self.space.extrinsic_to_intrinsic_coords(point_ext)
        result = self.space.intrinsic_to_extrinsic_coords(point_int)
        expected = point_ext
        expected = helper.to_vector(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

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
        base_point = tf.convert_to_tensor([1., 2., 3., 4., 6.])
        base_point = base_point / gs.linalg.norm(base_point)
        point = tf.convert_to_tensor([0., 5., 6., 2., -1.])
        point = point / gs.linalg.norm(point)

        log = self.metric.log(point=point, base_point=base_point)
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = point
        expected = helper.to_vector(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected), atol=1e-8)

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
        base_point = tf.convert_to_tensor([1., 2., 3., 4., 6.])
        base_point = base_point / gs.linalg.norm(base_point)
        point = (base_point
                 + 1e-12 * tf.convert_to_tensor([-1., -2., 1., 1., .1]))
        point = point / gs.linalg.norm(point)

        log = self.metric.log(point=point, base_point=base_point)
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = point
        expected = helper.to_vector(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_exp_vectorization(self):
        n_samples = self.n_samples
        dim = self.dimension + 1

        with self.test_session():
            one_vec = self.space.random_uniform()
            one_base_point = self.space.random_uniform()
            n_vecs = self.space.random_uniform(n_samples=n_samples)
            n_base_points = self.space.random_uniform(n_samples=n_samples)

            one_tangent_vec = self.space.projection_to_tangent_space(
                one_vec, base_point=one_base_point)
            result = self.metric.exp(one_tangent_vec, one_base_point)
            point_numpy = np.random.uniform(size=(1, dim))
            # TODO(nina): Fix that this test fails with assertShapeEqual
            self.assertAllClose(point_numpy.shape, gs.eval(gs.shape(result)))

        n_tangent_vecs = self.space.projection_to_tangent_space(
            n_vecs, base_point=one_base_point)
        result = self.metric.exp(n_tangent_vecs, one_base_point)
        point_numpy = np.random.uniform(size=(n_samples, dim))
        with self.test_session():
            self.assertShapeEqual(point_numpy, result)

        one_tangent_vec = self.space.projection_to_tangent_space(
            one_vec, base_point=n_base_points)
        result = self.metric.exp(one_tangent_vec, n_base_points)
        point_numpy = np.random.uniform(size=(n_samples, dim))
        with self.test_session():
            self.assertShapeEqual(point_numpy, result)

        n_tangent_vecs = self.space.projection_to_tangent_space(
            n_vecs, base_point=n_base_points)
        result = self.metric.exp(n_tangent_vecs, n_base_points)
        point_numpy = np.random.uniform(size=(n_samples, dim))
        with self.test_session():
            self.assertShapeEqual(point_numpy, result)

    def test_log_vectorization(self):
        n_samples = self.n_samples
        dim = self.dimension + 1

        one_base_point = self.space.random_uniform()
        one_point = self.space.random_uniform()
        n_points = self.space.random_uniform(n_samples=n_samples)
        n_base_points = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.log(one_point, one_base_point)
        point_numpy = np.random.uniform(size=(1, dim))
        with self.test_session():
            # TODO(nina): Fix that this test fails with assertShapeEqual
            self.assertAllClose(point_numpy.shape, gs.eval(gs.shape(result)))

        result = self.metric.log(n_points, one_base_point)
        point_numpy = np.random.uniform(size=(n_samples, dim))
        with self.test_session():
            self.assertShapeEqual(point_numpy, result)

        result = self.metric.log(one_point, n_base_points)
        point_numpy = np.random.uniform(size=(n_samples, dim))
        with self.test_session():
            self.assertShapeEqual(point_numpy, result)

        result = self.metric.log(n_points, n_base_points)
        point_numpy = np.random.uniform(size=(n_samples, dim))
        with self.test_session():
            self.assertShapeEqual(point_numpy, result)

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
        base_point = tf.convert_to_tensor([0., -3., 0., 3., 4.])
        base_point = base_point / gs.linalg.norm(base_point)
        vector = tf.convert_to_tensor([9., 5., 0., 0., -1.])
        vector = self.space.projection_to_tangent_space(
                                                   vector=vector,
                                                   base_point=base_point)

        exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
        result = self.metric.log(point=exp, base_point=base_point)

        expected = vector
        norm_expected = gs.linalg.norm(expected)
        regularized_norm_expected = gs.mod(norm_expected, 2 * gs.pi)
        expected = expected / norm_expected * regularized_norm_expected
        expected = helper.to_vector(expected)
        # TODO(nina): Fix that this test fails, in numpy
        # with self.test_session():
        #     self.assertAllClose(gs.eval(result), gs.eval(expected))

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
        base_point = tf.convert_to_tensor([10., -2., -.5, 34., 3.])
        base_point = base_point / gs.linalg.norm(base_point)
        vector = 1e-10 * tf.convert_to_tensor([.06, -51., 6., 5., 3.])
        vector = self.space.projection_to_tangent_space(
                                                    vector=vector,
                                                    base_point=base_point)

        exp = self.metric.exp(tangent_vec=vector, base_point=base_point)
        result = self.metric.log(point=exp, base_point=base_point)
        expected = self.space.projection_to_tangent_space(
                                                    vector=vector,
                                                    base_point=base_point)
        expected = helper.to_vector(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected), atol=1e-8)

    def test_squared_norm_and_squared_dist(self):
        """
        Test that the squared distance between two points is
        the squared norm of their logarithm.
        """
        point_a = (1. / gs.sqrt(129.)
                   * tf.convert_to_tensor([10., -2., -5., 0., 0.]))
        point_b = (1. / gs.sqrt(435.)
                   * tf.convert_to_tensor([1., -20., -5., 0., 3.]))
        log = self.metric.log(point=point_a, base_point=point_b)
        result = self.metric.squared_norm(vector=log)
        expected = self.metric.squared_dist(point_a, point_b)
        expected = helper.to_scalar(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_squared_dist_vectorization(self):
        n_samples = self.n_samples

        one_point_a = self.space.random_uniform()
        one_point_b = self.space.random_uniform()
        n_points_a = self.space.random_uniform(n_samples=n_samples)
        n_points_b = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.squared_dist(one_point_a, one_point_b)
        point_numpy = np.random.uniform(size=(1, 1))
        with self.test_session():
            # TODO(nina): Fix that this test fails with assertShapeEqual
            self.assertAllClose(point_numpy.shape, gs.eval(gs.shape(result)))

        result = self.metric.squared_dist(n_points_a, one_point_b)
        point_numpy = np.random.uniform(size=(n_samples, 1))
        with self.test_session():
            # TODO(nina): Fix that this test fails with assertShapeEqual
            self.assertAllClose(point_numpy.shape, gs.eval(gs.shape(result)))

        result = self.metric.squared_dist(one_point_a, n_points_b)
        point_numpy = np.random.uniform(size=(n_samples, 1))
        with self.test_session():
            # TODO(nina): Fix that this test fails with assertShapeEqual
            self.assertAllClose(point_numpy.shape, gs.eval(gs.shape(result)))

        result = self.metric.squared_dist(n_points_a, n_points_b)
        point_numpy = np.random.uniform(size=(n_samples, 1))
        with self.test_session():
            # TODO(nina): Fix that this test fails with assertShapeEqual
            self.assertAllClose(point_numpy.shape, gs.eval(gs.shape(result)))

    def test_norm_and_dist(self):
        """
        Test that the distance between two points is
        the norm of their logarithm.
        """
        point_a = (1. / gs.sqrt(129.)
                   * tf.convert_to_tensor([10., -2., -5., 0., 0.]))
        point_b = (1. / gs.sqrt(435.)
                   * tf.convert_to_tensor([1., -20., -5., 0., 3.]))
        log = self.metric.log(point=point_a, base_point=point_b)
        result = self.metric.norm(vector=log)
        expected = self.metric.dist(point_a, point_b)
        expected = helper.to_scalar(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_dist_point_and_itself(self):
        # Distance between a point and itself is 0
        point_a = (1. / gs.sqrt(129.)
                   * tf.convert_to_tensor([10., -2., -5., 0., 0.]))
        point_b = point_a
        result = self.metric.dist(point_a, point_b)
        expected = 0.
        expected = helper.to_scalar(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_dist_orthogonal_points(self):
        # Distance between two orthogonal points is pi / 2.
        point_a = gs.array([10., -2., -.5, 0., 0.])
        point_a = point_a / gs.linalg.norm(point_a)
        point_b = gs.array([2., 10, 0., 0., 0.])
        point_b = point_b / gs.linalg.norm(point_b)
        result = gs.dot(point_a, point_b)
        result = helper.to_scalar(result)
        expected = 0
        expected = helper.to_scalar(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        result = self.metric.dist(point_a, point_b)
        expected = gs.pi / 2
        expected = helper.to_scalar(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_exp_and_dist_and_projection_to_tangent_space(self):
        with self.test_session():
            base_point = tf.convert_to_tensor([16., -2., -2.5, 84., 3.])
            base_point = base_point / gs.linalg.norm(base_point)
            vector = tf.convert_to_tensor([9., 0., -1., -2., 1.])
            tangent_vec = self.space.projection_to_tangent_space(
                                                      vector=vector,
                                                      base_point=base_point)

            exp = self.metric.exp(tangent_vec=tangent_vec,
                                  base_point=base_point)
            result = self.metric.dist(base_point, exp)
            expected = gs.linalg.norm(tangent_vec) % (2 * gs.pi)
            expected = helper.to_scalar(expected)
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_exp_and_dist_and_projection_to_tangent_space_vec(self):
        with self.test_session():

            base_point = tf.convert_to_tensor([
                [16., -2., -2.5, 84., 3.],
                [16., -2., -2.5, 84., 3.]])

            base_single_point = tf.convert_to_tensor([16., -2., -2.5, 84., 3.])
            scalar_norm = gs.linalg.norm(base_single_point)

            base_point = base_point / scalar_norm
            vector = tf.convert_to_tensor(
                    [[9., 0., -1., -2., 1.],
                     [9., 0., -1., -2., 1]])

            tangent_vec = self.space.projection_to_tangent_space(
                    vector=vector,
                    base_point=base_point)

            exp = self.metric.exp(tangent_vec=tangent_vec,
                                  base_point=base_point)

            result = self.metric.dist(base_point, exp)
            expected = gs.linalg.norm(tangent_vec, axis=-1) % (2 * gs.pi)

            expected = helper.to_scalar(expected)
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_geodesic_and_belongs(self):
        n_geodesic_points = 100
        initial_point = self.space.random_uniform()
        vector = tf.convert_to_tensor([2., 0., -1., -2., 1.])
        initial_tangent_vec = self.space.projection_to_tangent_space(
                                            vector=vector,
                                            base_point=initial_point)
        geodesic = self.metric.geodesic(
                                   initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)

        t = gs.linspace(start=0., stop=1., num=n_geodesic_points)
        points = geodesic(t)

        bool_belongs = self.space.belongs(points)
        expected = tf.convert_to_tensor(n_geodesic_points * [[True]])

        with self.test_session():
            self.assertAllClose(gs.eval(expected), gs.eval(bool_belongs))


if __name__ == '__main__':
    tf.test.main()
