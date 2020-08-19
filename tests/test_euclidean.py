"""Unit tests for the Euclidean space."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean


class TestEuclidean(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.dimension = 2
        self.space = Euclidean(self.dimension)
        self.metric = self.space.metric

        self.n_samples = 3

        self.one_point_a = gs.array([0., 1.])
        self.one_point_b = gs.array([2., 10.])
        self.n_points_a = gs.array([
            [2., 1.],
            [-2., -4.],
            [-5., 1.]])
        self.n_points_b = gs.array([
            [2., 10.],
            [8., -1.],
            [-3., 6.]])

    def test_random_uniform_and_belongs(self):
        point = self.space.random_uniform()
        result = self.space.belongs(point)
        expected = True

        self.assertAllClose(result, expected)

    def test_squared_norm_vectorization(self):
        n_samples = self.n_samples
        n_points = gs.array([
            [2., 1.],
            [-2., -4.],
            [-5., 1.]])
        result = self.metric.squared_norm(n_points)

        expected = gs.array([5., 20., 26.])

        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

    def test_norm_vectorization_single_sample(self):
        one_point = gs.array([[0., 1.]])

        result = self.metric.norm(one_point)
        expected = gs.array([1.])
        self.assertAllClose(gs.shape(result), (1,))
        self.assertAllClose(result, expected)

        one_point = gs.array([0., 1.])

        result = self.metric.norm(one_point)
        expected = 1.
        self.assertAllClose(gs.shape(result), ())
        self.assertAllClose(result, expected)

    def test_norm_vectorization_n_samples(self):
        n_samples = self.n_samples
        n_points = gs.array([
            [2., 1.],
            [-2., -4.],
            [-5., 1.]])

        result = self.metric.norm(n_points)

        expected = gs.array([2.2360679775, 4.472135955, 5.09901951359])

        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

    def test_exp_vectorization(self):
        n_samples = self.n_samples
        dim = self.dimension

        one_tangent_vec = gs.array([0., 1.])
        one_base_point = gs.array([2., 10.])
        n_tangent_vecs = gs.array([
            [2., 1.],
            [-2., -4.],
            [-5., 1.]])
        n_base_points = gs.array([
            [2., 10.],
            [8., -1.],
            [-3., 6.]])

        result = self.metric.exp(one_tangent_vec, one_base_point)
        expected = one_tangent_vec + one_base_point

        self.assertAllClose(result, expected)

        result = self.metric.exp(n_tangent_vecs, one_base_point)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.exp(one_tangent_vec, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.exp(n_tangent_vecs, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

    def test_log_vectorization(self):
        n_samples = self.n_samples
        dim = self.dimension

        one_point = gs.array([0., 1.])
        one_base_point = gs.array([2., 10.])
        n_points = gs.array([
            [2., 1.],
            [-2., -4.],
            [-5., 1.]])
        n_base_points = gs.array([
            [2., 10.],
            [8., -1.],
            [-3., 6.]])

        result = self.metric.log(one_point, one_base_point)
        expected = one_point - one_base_point
        self.assertAllClose(result, expected)

        result = self.metric.log(n_points, one_base_point)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.log(one_point, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

        result = self.metric.log(n_points, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, dim))

    def test_squared_dist_vectorization(self):
        n_samples = self.n_samples

        one_point_a = gs.array([0., 1.])
        one_point_b = gs.array([2., 10.])
        n_points_a = gs.array([
            [2., 1.],
            [-2., -4.],
            [-5., 1.]])
        n_points_b = gs.array([
            [2., 10.],
            [8., -1.],
            [-3., 6.]])

        result = self.metric.squared_dist(one_point_a, one_point_b)
        vec = one_point_a - one_point_b
        expected = gs.dot(vec, gs.transpose(vec))
        self.assertAllClose(result, expected)

        result = self.metric.squared_dist(n_points_a, one_point_b)
        self.assertAllClose(gs.shape(result), (n_samples,))

        result = self.metric.squared_dist(one_point_a, n_points_b)
        self.assertAllClose(gs.shape(result), (n_samples,))

        result = self.metric.squared_dist(n_points_a, n_points_b)
        expected = gs.array([81., 109., 29.])
        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

    def test_dist_vectorization(self):
        n_samples = self.n_samples

        one_point_a = gs.array([0., 1.])
        one_point_b = gs.array([2., 10.])
        n_points_a = gs.array([
            [2., 1.],
            [-2., -4.],
            [-5., 1.]])
        n_points_b = gs.array([
            [2., 10.],
            [8., -1.],
            [-3., 6.]])

        result = self.metric.dist(one_point_a, one_point_b)
        vec = one_point_a - one_point_b
        expected = gs.sqrt(gs.dot(vec, gs.transpose(vec)))
        self.assertAllClose(result, expected)

        result = self.metric.dist(n_points_a, one_point_b)
        self.assertAllClose(gs.shape(result), (n_samples,))

        result = self.metric.dist(one_point_a, n_points_b)
        self.assertAllClose(gs.shape(result), (n_samples,))

        result = self.metric.dist(n_points_a, n_points_b)
        expected = gs.array([9., gs.sqrt(109.), gs.sqrt(29.)])

        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

    def test_belongs(self):
        point = gs.array([0., 1.])

        result = self.space.belongs(point)
        expected = True

        self.assertAllClose(result, expected)

    def test_random_uniform(self):
        result = self.space.random_uniform()

        self.assertAllClose(gs.shape(result), (self.dimension,))

    def test_inner_product_matrix(self):
        result = self.metric.inner_product_matrix()

        expected = gs.eye(self.dimension)

        self.assertAllClose(result, expected)

    def test_inner_product(self):
        point_a = gs.array([0., 1.])
        point_b = gs.array([2., 10.])

        result = self.metric.inner_product(point_a, point_b)
        expected = 10.

        self.assertAllClose(result, expected)

    def test_inner_product_vectorization_single_sample(self):
        one_point_a = gs.array([[0., 1.]])
        one_point_b = gs.array([[2., 10.]])

        result = self.metric.inner_product(one_point_a, one_point_b)
        expected = gs.array([10.])
        self.assertAllClose(gs.shape(result), (1,))
        self.assertAllClose(result, expected)

        one_point_a = gs.array([[0., 1.]])
        one_point_b = gs.array([2., 10.])

        result = self.metric.inner_product(one_point_a, one_point_b)
        expected = gs.array([10.])
        self.assertAllClose(gs.shape(result), (1,))
        self.assertAllClose(result, expected)

        one_point_a = gs.array([0., 1.])
        one_point_b = gs.array([[2., 10.]])

        result = self.metric.inner_product(one_point_a, one_point_b)
        expected = gs.array([10.])
        self.assertAllClose(gs.shape(result), (1,))
        self.assertAllClose(result, expected)

        one_point_a = gs.array([0., 1.])
        one_point_b = gs.array([2., 10.])

        result = self.metric.inner_product(one_point_a, one_point_b)
        expected = 10.
        self.assertAllClose(gs.shape(result), ())
        self.assertAllClose(result, expected)

    def test_inner_product_vectorization_n_samples(self):
        n_samples = 3
        n_points_a = gs.array([
            [2., 1.],
            [-2., -4.],
            [-5., 1.]])
        n_points_b = gs.array([
            [2., 10.],
            [8., -1.],
            [-3., 6.]])

        one_point_a = gs.array([0., 1.])
        one_point_b = gs.array([2., 10.])

        result = self.metric.inner_product(n_points_a, one_point_b)
        expected = gs.array([14., -44., 0.])
        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

        result = self.metric.inner_product(one_point_a, n_points_b)
        expected = gs.array([10., -1., 6.])
        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

        result = self.metric.inner_product(n_points_a, n_points_b)
        expected = gs.array([14., -12., 21.])
        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

        one_point_a = gs.array([[0., 1.]])
        one_point_b = gs.array([[2., 10]])

        result = self.metric.inner_product(n_points_a, one_point_b)
        expected = gs.array([14., -44., 0.])
        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

        result = self.metric.inner_product(one_point_a, n_points_b)
        expected = gs.array([10., -1., 6.])
        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

        result = self.metric.inner_product(n_points_a, n_points_b)
        expected = gs.array([14., -12., 21.])
        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

        one_point_a = gs.array([[0., 1.]])
        one_point_b = gs.array([2., 10.])

        result = self.metric.inner_product(n_points_a, one_point_b)
        expected = gs.array([14., -44., 0.])
        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

        result = self.metric.inner_product(one_point_a, n_points_b)
        expected = gs.array([10., -1., 6.])
        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

        result = self.metric.inner_product(n_points_a, n_points_b)
        expected = gs.array([14., -12., 21.])
        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

        one_point_a = gs.array([0., 1.])
        one_point_b = gs.array([[2., 10.]])

        result = self.metric.inner_product(n_points_a, one_point_b)
        expected = gs.array([14., -44., 0.])
        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

        result = self.metric.inner_product(one_point_a, n_points_b)
        expected = gs.array([10., -1., 6.])
        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

        result = self.metric.inner_product(n_points_a, n_points_b)
        expected = gs.array([14., -12., 21.])
        self.assertAllClose(gs.shape(result), (n_samples,))
        self.assertAllClose(result, expected)

    def test_squared_norm(self):
        point = gs.array([-2., 4.])

        result = self.metric.squared_norm(point)
        expected = 20.

        self.assertAllClose(result, expected)

    def test_norm(self):
        point = gs.array([-2., 4.])
        result = self.metric.norm(point)
        expected = 4.472135955

        self.assertAllClose(result, expected)

    def test_exp(self):
        base_point = gs.array([0., 1.])
        vector = gs.array([2., 10.])

        result = self.metric.exp(tangent_vec=vector,
                                 base_point=base_point)
        expected = base_point + vector

        self.assertAllClose(result, expected)

    def test_log(self):
        base_point = gs.array([0., 1.])
        point = gs.array([2., 10.])

        result = self.metric.log(point=point, base_point=base_point)
        expected = point - base_point

        self.assertAllClose(result, expected)

    def test_squared_dist(self):
        point_a = gs.array([-1., 4.])
        point_b = gs.array([1., 1.])

        result = self.metric.squared_dist(point_a, point_b)
        vec = point_b - point_a
        expected = gs.dot(vec, vec)

        self.assertAllClose(result, expected)

    def test_dist(self):
        point_a = gs.array([0., 1.])
        point_b = gs.array([2., 10.])

        result = self.metric.dist(point_a, point_b)
        expected = gs.linalg.norm(point_b - point_a)

        self.assertAllClose(result, expected)

    def test_geodesic_and_belongs(self):
        n_geodesic_points = 100
        initial_point = gs.array([[2., -1.]])
        initial_tangent_vec = gs.array([2., 0.])
        geodesic = self.metric.geodesic(
            initial_point=initial_point,
            initial_tangent_vec=initial_tangent_vec)

        t = gs.linspace(start=0., stop=1., num=n_geodesic_points)
        points = geodesic(t)

        result = self.space.belongs(points)
        expected = gs.array(n_geodesic_points * [True])

        self.assertAllClose(expected, result)
