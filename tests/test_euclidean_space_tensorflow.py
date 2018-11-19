"""
Unit tests for Euclidean Space for tensorflow backend.
"""

import importlib
import numpy as np
import os
import tensorflow as tf

import geomstats.backend as gs
import tests.helper as helper

from geomstats.euclidean_space import EuclideanSpace


class TestEuclideanSpaceMethodsTensorFlow(tf.test.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)

        self.dimension = 2
        self.space = EuclideanSpace(self.dimension)
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
        belongs = self.space.belongs(point)
        expected = tf.convert_to_tensor([[True]])

        with self.test_session():
            self.assertAllClose(gs.eval(belongs), gs.eval(expected))

    def test_random_uniform(self):
        point = self.space.random_uniform()
        point_numpy = np.random.uniform(size=(1, self.dimension))

        with self.test_session():
            self.assertShapeEqual(point_numpy, point)

    def test_inner_product_matrix(self):
        result = self.metric.inner_product_matrix()

        expected = gs.eye(self.dimension)
        expected = helper.to_matrix(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_inner_product(self):
        point_a = tf.convert_to_tensor([0., 1.])
        point_b = tf.convert_to_tensor([2., 10.])

        result = self.metric.inner_product(point_a, point_b)
        expected = gs.dot(point_a, point_b)
        expected = helper.to_scalar(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_inner_product_vectorization(self):
        n_samples = 3

        one_point_a = tf.convert_to_tensor([0., 1.])
        one_point_b = tf.convert_to_tensor([2., 10.])

        n_points_a = tf.convert_to_tensor([
            [2., 1.],
            [-2., -4.],
            [-5., 1.]])
        n_points_b = tf.convert_to_tensor([
            [2., 10.],
            [8., -1.],
            [-3., 6.]])

        result = self.metric.inner_product(one_point_a, one_point_b)
        expected = gs.dot(one_point_a, gs.transpose(one_point_b))
        expected = helper.to_scalar(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        result = self.metric.inner_product(n_points_a, one_point_b)
        point_numpy = np.random.uniform(size=(n_samples, 1))
        # TODO(nina): Fix this test with assertShapeEqual
        with self.test_session():
            self.assertAllClose(point_numpy.shape, gs.eval(gs.shape(result)))

        result = self.metric.inner_product(one_point_a, n_points_b)
        point_numpy = np.random.uniform(size=(n_samples, 1))
        # TODO(nina): Fix this test with assertShapeEqual
        with self.test_session():
            self.assertAllClose(point_numpy.shape, gs.eval(gs.shape(result)))

        result = self.metric.inner_product(n_points_a, n_points_b)
        point_numpy = np.random.uniform(size=(n_samples, 1))
        # TODO(nina): Fix this test with assertShapeEqual
        with self.test_session():
            self.assertAllClose(point_numpy.shape, gs.eval(gs.shape(result)))

    def test_squared_norm(self):
        point = tf.convert_to_tensor([-2., 4.])

        result = self.metric.squared_norm(point)
        expected = gs.linalg.norm(point) ** 2
        expected = helper.to_scalar(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_norm(self):
        point = tf.convert_to_tensor([-2., 4.])

        result = self.metric.norm(point)
        expected = gs.linalg.norm(point)
        expected = helper.to_scalar(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_exp(self):
        base_point = tf.convert_to_tensor([0., 1.])
        vector = tf.convert_to_tensor([2., 10.])

        result = self.metric.exp(tangent_vec=vector,
                                 base_point=base_point)
        expected = base_point + vector
        expected = helper.to_vector(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_log(self):
        base_point = tf.convert_to_tensor([0., 1.])
        point = tf.convert_to_tensor([2., 10.])

        result = self.metric.log(point=point, base_point=base_point)
        expected = point - base_point
        expected = helper.to_vector(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_squared_dist(self):
        point_a = tf.convert_to_tensor([-1., 4.])
        point_b = tf.convert_to_tensor([1., 1.])

        result = self.metric.squared_dist(point_a, point_b)
        vec = point_b - point_a
        expected = gs.dot(vec, vec)
        expected = helper.to_scalar(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_dist(self):
        point_a = tf.convert_to_tensor([0., 1.])
        point_b = tf.convert_to_tensor([2., 10.])

        result = self.metric.dist(point_a, point_b)
        expected = gs.linalg.norm(point_b - point_a)
        expected = helper.to_scalar(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_geodesic_and_belongs(self):
        n_geodesic_points = 100
        initial_point = self.space.random_uniform()
        initial_tangent_vec = tf.convert_to_tensor([2., 0.])
        geodesic = self.metric.geodesic(
                                   initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)

        t = gs.linspace(start=0., stop=1., num=n_geodesic_points)
        points = geodesic(t)

        bool_belongs = self.space.belongs(points)
        expected = tf.convert_to_tensor(n_geodesic_points * [[True]])

        with self.test_session():
            self.assertAllClose(gs.eval(expected), gs.eval(bool_belongs))

    def test_mean(self):
        # TODO(nina): Fix the fact that it doesn't work for [1., 4.]
        point = tf.convert_to_tensor([[1., 4.]])
        result = self.metric.mean(points=[point, point, point])
        expected = point
        expected = helper.to_vector(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        points = tf.convert_to_tensor([
            [1., 2.],
            [2., 3.],
            [3., 4.],
            [4., 5.]])
        weights = gs.array([1., 2., 1., 2.])

        result = self.metric.mean(points, weights)
        expected = tf.convert_to_tensor([16. / 6., 22. / 6.])
        expected = helper.to_vector(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_variance(self):
        points = tf.convert_to_tensor([
            [1., 2.],
            [2., 3.],
            [3., 4.],
            [4., 5.]])
        weights = tf.convert_to_tensor([1., 2., 1., 2.])
        base_point = gs.zeros(2)
        result = self.metric.variance(points, weights, base_point)
        # we expect the average of the points' sq norms.
        expected = (1 * 5. + 2 * 13. + 1 * 25. + 2 * 41.) / 6.
        expected = helper.to_scalar(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))


if __name__ == '__main__':
        tf.test.main()
