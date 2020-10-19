"""Unit tests for Minkowski space."""

import math

import numpy as np

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.minkowski import Minkowski


class TestMinkowski(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.time_like_dim = 0
        self.dimension = 2
        self.space = Minkowski(self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10

    def test_belongs(self):
        point = gs.array([-1., 3.])
        result = self.space.belongs(point)
        expected = True

        self.assertAllClose(result, expected)

    def test_random_uniform(self):
        point = self.space.random_uniform()
        self.assertAllClose(gs.shape(point), (self.dimension,))

    def test_random_uniform_and_belongs(self):
        point = self.space.random_uniform()
        result = self.space.belongs(point)
        expected = True
        self.assertAllClose(result, expected)

    def test_inner_product_matrix(self):
        result = self.metric.inner_product_matrix()

        expected = gs.array([[-1., 0.], [0., 1.]])
        self.assertAllClose(result, expected)

    def test_inner_product(self):
        point_a = gs.array([0., 1.])
        point_b = gs.array([2., 10.])

        result = self.metric.inner_product(point_a, point_b)
        expected = gs.dot(point_a, point_b)
        expected -= (2 * point_a[self.time_like_dim]
                     * point_b[self.time_like_dim])

        self.assertAllClose(result, expected)

    def test_inner_product_vectorization(self):
        n_samples = 3
        one_point_a = gs.array([-1., 0.])
        one_point_b = gs.array([1.0, 0.])

        n_points_a = gs.array([
            [-1., 0.],
            [1., 0.],
            [2., math.sqrt(3)]])
        n_points_b = gs.array([
            [2., -math.sqrt(3)],
            [4.0, math.sqrt(15)],
            [-4.0, math.sqrt(15)]])

        result = self.metric.inner_product(one_point_a, one_point_b)
        expected = gs.dot(one_point_a, gs.transpose(one_point_b))
        expected -= (2 * one_point_a[self.time_like_dim]
                     * one_point_b[self.time_like_dim])

        result_no = self.metric.inner_product(n_points_a,
                                              one_point_b)
        result_on = self.metric.inner_product(one_point_a, n_points_b)

        result_nn = self.metric.inner_product(n_points_a, n_points_b)

        self.assertAllClose(result, expected)
        self.assertAllClose(gs.shape(result_no), (n_samples,))
        self.assertAllClose(gs.shape(result_on), (n_samples,))
        self.assertAllClose(gs.shape(result_nn), (n_samples,))

        expected = np.zeros(n_samples)
        for i in range(n_samples):
            expected[i] = gs.dot(n_points_a[i], n_points_b[i])
            expected[i] -= (2 * n_points_a[i, self.time_like_dim]
                            * n_points_b[i, self.time_like_dim])

        self.assertAllClose(result_nn, expected)

    def test_squared_norm(self):
        point = gs.array([-2., 4.])

        result = self.metric.squared_norm(point)
        expected = 12.
        self.assertAllClose(result, expected)

    def test_squared_norm_vectorization(self):
        n_samples = 3
        n_points = gs.array([
            [-1., 0.],
            [1., 0.],
            [2., math.sqrt(3)]])

        result = self.metric.squared_norm(n_points)
        self.assertAllClose(gs.shape(result), (n_samples,))

    def test_exp(self):
        base_point = gs.array([1.0, 0.])
        vector = gs.array([2., math.sqrt(3)])

        result = self.metric.exp(tangent_vec=vector,
                                 base_point=base_point)
        expected = base_point + vector
        self.assertAllClose(result, expected)

    def test_exp_vectorization(self):
        dim = self.dimension
        n_samples = 3
        one_tangent_vec = gs.array([-1., 0.])
        one_base_point = gs.array([1.0, 0.])

        n_tangent_vecs = gs.array([
            [-1., 0.],
            [1., 0.],
            [2., math.sqrt(3)]])
        n_base_points = gs.array([
            [2., -math.sqrt(3)],
            [4.0, math.sqrt(15)],
            [-4.0, math.sqrt(15)]])

        result = self.metric.exp(one_tangent_vec, one_base_point)
        expected = one_tangent_vec + one_base_point
        self.assertAllClose(result, expected)

        result = self.metric.exp(n_tangent_vecs, one_base_point)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.exp(one_tangent_vec, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.exp(n_tangent_vecs, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

    def test_log(self):
        base_point = gs.array([-1., 0.])
        point = gs.array([2., math.sqrt(3)])

        result = self.metric.log(point=point, base_point=base_point)
        expected = point - base_point
        self.assertAllClose(result, expected)

    def test_log_vectorization(self):
        dim = self.dimension
        n_samples = 3
        one_point = gs.array([-1., 0.])
        one_base_point = gs.array([1.0, 0.])

        n_points = gs.array([
            [-1., 0.],
            [1., 0.],
            [2., math.sqrt(3)]])
        n_base_points = gs.array([
            [2., -math.sqrt(3)],
            [4.0, math.sqrt(15)],
            [-4.0, math.sqrt(15)]])

        result = self.metric.log(one_point, one_base_point)
        expected = one_point - one_base_point
        self.assertAllClose(result, expected)

        result = self.metric.log(n_points, one_base_point)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.log(one_point, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.log(n_points, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

    def test_squared_dist(self):
        point_a = gs.array([2., -math.sqrt(3)])
        point_b = gs.array([4.0, math.sqrt(15)])

        result = self.metric.squared_dist(point_a, point_b)
        vec = point_b - point_a
        expected = gs.dot(vec, vec)
        expected -= 2 * vec[self.time_like_dim] * vec[self.time_like_dim]
        self.assertAllClose(result, expected)

    def test_geodesic_and_belongs(self):
        n_geodesic_points = 100
        initial_point = gs.array([2., -math.sqrt(3)])
        initial_tangent_vec = gs.array([2., 0.])

        geodesic = self.metric.geodesic(
            initial_point=initial_point,
            initial_tangent_vec=initial_tangent_vec)

        t = gs.linspace(start=0., stop=1., num=n_geodesic_points)
        points = geodesic(t)

        result = self.space.belongs(points)
        expected = gs.array(n_geodesic_points * [True])

        self.assertAllClose(result, expected)
