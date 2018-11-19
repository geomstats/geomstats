"""
Unit tests for Minkowski space with tensorflow backend.
"""

import importlib
import math
import numpy as np
import os
import tensorflow as tf

import geomstats.backend as gs
import tests.helper as helper

from geomstats.minkowski_space import MinkowskiSpace


class TestMinkowskiSpaceTensorFlow(tf.test.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)

        self.time_like_dim = 0
        self.dimension = 2
        self.space = MinkowskiSpace(self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10

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
        with self.test_session():
            self.assertAllClose(gs.eval(point).shape, (1, self.dimension))

    def test_random_uniform_and_belongs(self):
        point = self.space.random_uniform()
        with self.test_session():
            self.assertTrue(gs.eval(self.space.belongs(point)))

    def test_inner_product_matrix(self):
        result = self.metric.inner_product_matrix()

        expected = tf.convert_to_tensor([[-1.0, 0.], [0., 1.]])
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_inner_product(self):
        point_a = tf.convert_to_tensor([0., 1.])
        point_b = tf.convert_to_tensor([2., 10.])

        result = self.metric.inner_product(point_a, point_b)
        expected = helper.to_scalar(gs.dot(point_a, point_b))
        expected -= (2 * point_a[self.time_like_dim]
                     * point_b[self.time_like_dim])

        with self.test_session():

            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_inner_product_vectorization(self):
        n_samples = 3
        one_point_a = tf.convert_to_tensor([[-1., 0.]])
        one_point_b = tf.convert_to_tensor([[1.0, 0.]])

        n_points_a = tf.convert_to_tensor([
            [-1., 0.],
            [1., 0.],
            [2., math.sqrt(3)]])
        n_points_b = tf.convert_to_tensor([
            [2., -math.sqrt(3)],
            [4.0, math.sqrt(15)],
            [-4.0, math.sqrt(15)]])

        result = self.metric.inner_product(one_point_a, one_point_b)
        expected = gs.dot(one_point_a, gs.transpose(one_point_b))
        expected -= (2 * one_point_a[:, self.time_like_dim]
                     * one_point_b[:, self.time_like_dim])
        expected = helper.to_scalar(expected)

        result_no = self.metric.inner_product(n_points_a,
                                              one_point_b)
        result_on = self.metric.inner_product(one_point_a, n_points_b)

        result_nn = self.metric.inner_product(n_points_a, n_points_b)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))
            self.assertAllClose(gs.eval(result_no).shape,
                                (n_samples, 1))
            self.assertAllClose(gs.eval(result_on).shape,
                                (n_samples, 1))
            self.assertAllClose(gs.eval(result_nn).shape, (n_samples, 1))

            expected = np.zeros(n_samples)
            for i in range(n_samples):
                expected[i] = gs.eval(gs.dot(n_points_a[i],
                                             n_points_b[i])
                                      )
                expected[i] -= (2 * gs.eval(n_points_a[i, self.time_like_dim])
                                * gs.eval(n_points_b[i, self.time_like_dim]))
            expected = helper.to_scalar(tf.convert_to_tensor(expected))

            self.assertAllClose(gs.eval(result_nn), gs.eval(expected))

    def test_squared_norm(self):
        point = tf.convert_to_tensor([-2., 4.])

        result = self.metric.squared_norm(point)
        expected = gs.dot(point, point)
        expected -= 2 * point[self.time_like_dim] * point[self.time_like_dim]
        expected = helper.to_scalar(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_squared_norm_vectorization(self):
        n_samples = 3
        n_points = tf.convert_to_tensor([
            [-1., 0.],
            [1., 0.],
            [2., math.sqrt(3)]])

        result = self.metric.squared_norm(n_points)
        with self.test_session():
            self.assertAllClose(gs.eval(result).shape, (n_samples, 1))

    def test_exp(self):
        base_point = tf.convert_to_tensor([1.0, 0.])
        vector = tf.convert_to_tensor([2., math.sqrt(3)])

        result = self.metric.exp(tangent_vec=vector,
                                 base_point=base_point)
        expected = base_point + vector
        expected = helper.to_vector(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_exp_vectorization(self):
        dim = self.dimension
        n_samples = 3
        one_tangent_vec = tf.convert_to_tensor([[-1., 0.]])
        one_base_point = tf.convert_to_tensor([[1.0, 0.]])

        n_tangent_vecs = tf.convert_to_tensor([
            [-1., 0.],
            [1., 0.],
            [2., math.sqrt(3)]])
        n_base_points = tf.convert_to_tensor([
            [2., -math.sqrt(3)],
            [4.0, math.sqrt(15)],
            [-4.0, math.sqrt(15)]])

        result = self.metric.exp(one_tangent_vec, one_base_point)
        expected = one_tangent_vec + one_base_point
        expected = helper.to_vector(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

            result = self.metric.exp(n_tangent_vecs, one_base_point)
            self.assertAllClose(gs.eval(result).shape, (n_samples, dim))

            result = self.metric.exp(one_tangent_vec, n_base_points)
            self.assertAllClose(gs.eval(result).shape, (n_samples, dim))

            result = self.metric.exp(n_tangent_vecs, n_base_points)
            self.assertAllClose(gs.eval(result).shape, (n_samples, dim))

    def test_log(self):
        base_point = tf.convert_to_tensor([-1., 0.])
        point = tf.convert_to_tensor([2., math.sqrt(3)])

        result = self.metric.log(point=point, base_point=base_point)
        expected = point - base_point
        expected = helper.to_vector(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_log_vectorization(self):

        dim = self.dimension
        n_samples = 3
        one_point = tf.convert_to_tensor([[-1., 0.]])
        one_base_point = tf.convert_to_tensor([[1.0, 0.]])

        n_points = tf.convert_to_tensor([
            [-1., 0.],
            [1., 0.],
            [2., math.sqrt(3)]])
        n_base_points = tf.convert_to_tensor([
            [2., -math.sqrt(3)],
            [4.0, math.sqrt(15)],
            [-4.0, math.sqrt(15)]])

        result = self.metric.log(one_point, one_base_point)
        expected = one_point - one_base_point
        expected = helper.to_vector(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

            result = self.metric.log(n_points, one_base_point)
            self.assertAllClose(gs.eval(result).shape, (n_samples, dim))

            result = self.metric.log(one_point, n_base_points)
            self.assertAllClose(gs.eval(result).shape, (n_samples, dim))

            result = self.metric.log(n_points, n_base_points)
            self.assertAllClose(gs.eval(result).shape, (n_samples, dim))

    def test_squared_dist(self):
        point_a = tf.convert_to_tensor([2., -math.sqrt(3)])
        point_b = tf.convert_to_tensor([4.0, math.sqrt(15)])

        result = self.metric.squared_dist(point_a, point_b)
        vec = point_b - point_a
        expected = gs.dot(vec, vec)
        expected -= 2 * vec[self.time_like_dim] * vec[self.time_like_dim]
        expected = helper.to_scalar(expected)
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_geodesic_and_belongs(self):
        n_geodesic_points = 100
        initial_point = tf.convert_to_tensor([2., -math.sqrt(3)])
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
        point = tf.convert_to_tensor([[2., -math.sqrt(3)]])
        result = self.metric.mean(points=[point, point, point])
        expected = point
        expected = helper.to_vector(expected)

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

        points = tf.convert_to_tensor([
            [1., 0.],
            [2., math.sqrt(3)],
            [3., math.sqrt(8)],
            [4., math.sqrt(24)]])
        weights = gs.array([1., 2., 1., 2.])
        result = self.metric.mean(points, weights)
        result = self.space.belongs(result)
        expected = tf.convert_to_tensor([[True]])
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_variance(self):
        points = tf.convert_to_tensor([
            [1., 0.],
            [2., math.sqrt(3)],
            [3., math.sqrt(8)],
            [4., math.sqrt(24)]])
        weights = tf.convert_to_tensor([1., 2., 1., 2.])
        base_point = tf.convert_to_tensor([-1., 0.])
        result = self.metric.variance(points, weights, base_point)
        # we expect the average of the points' Minkowski sq norms.
        expected = helper.to_scalar(tf.convert_to_tensor([True]))
        with self.test_session():
            self.assertAllClose(gs.eval(result) != 0, gs.eval(expected))


if __name__ == '__main__':
    tf.test.main()
