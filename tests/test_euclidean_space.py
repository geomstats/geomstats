"""Unit tests for euclidean space module."""

import numpy as np
import unittest

from geomstats.euclidean_space import EuclideanSpace


class TestEuclideanSpaceMethods(unittest.TestCase):
    def setUp(self):
        self.dimension = 2
        self.space = EuclideanSpace(self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10

    def test_inner_product_matrix(self):
        result = self.metric.inner_product_matrix()

        expected = np.eye(self.dimension)
        self.assertTrue(np.allclose(result, expected))

    def test_inner_product(self):
        point_a = np.array([0, 1])
        point_b = np.array([2, 10])

        result = self.metric.inner_product(point_a, point_b)
        expected = np.dot(point_a, point_b)
        self.assertTrue(np.allclose(result, expected))

    def test_inner_product_vectorization(self):
        n_samples = self.n_samples
        one_point_a = self.space.random_uniform(n_samples=1)
        one_point_b = self.space.random_uniform(n_samples=1)
        n_points_a = self.space.random_uniform(n_samples=n_samples)
        n_points_b = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.inner_product(one_point_a, one_point_b)
        expected = np.dot(one_point_a, one_point_b.transpose())
        self.assertTrue(np.allclose(result, expected))

        result = self.metric.inner_product(n_points_a, one_point_b)
        expected = np.dot(n_points_a, one_point_b.transpose())
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))
        self.assertTrue(np.allclose(result, expected))

        result = self.metric.inner_product(one_point_a, n_points_b)
        expected = np.dot(one_point_a, n_points_b.transpose()).transpose()
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))
        self.assertTrue(np.allclose(result, expected))

        result = self.metric.inner_product(n_points_a, n_points_b)
        expected = np.zeros((n_samples, 1))
        for i in range(n_samples):
            expected[i] = np.dot(n_points_a[i], n_points_b[i])
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))
        self.assertTrue(np.allclose(result, expected))

    def test_squared_norm(self):
        point = np.array([-2, 4])

        result = self.metric.squared_norm(point)
        expected = np.linalg.norm(point) ** 2
        self.assertTrue(np.allclose(result, expected))

    def test_squared_norm_vectorization(self):
        n_samples = self.n_samples
        n_points = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.squared_norm(n_points)
        expected = np.linalg.norm(n_points, axis=1) ** 2
        expected = np.expand_dims(expected, axis=1)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))
        self.assertTrue(np.allclose(result, expected))

    def test_norm(self):
        point = np.array([-2, 4])

        result = self.metric.norm(point)
        expected = np.linalg.norm(point)
        self.assertTrue(np.allclose(result, expected))

    def test_norm_vectorization(self):
        n_samples = self.n_samples
        n_points = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.norm(n_points)
        expected = np.linalg.norm(n_points, axis=1)
        expected = np.expand_dims(expected, axis=1)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))
        self.assertTrue(np.allclose(result, expected),
                        '\n result = {}'
                        '\n expected = {}'.format(result, expected))

    def test_exp(self):
        base_point = np.array([0, 1])
        vector = np.array([2, 10])

        result = self.metric.exp(tangent_vec=vector,
                                 base_point=base_point)
        expected = base_point + vector
        self.assertTrue(np.allclose(result, expected))

    def test_exp_vectorization(self):
        n_samples = self.n_samples
        dim = self.dimension
        one_tangent_vec = self.space.random_uniform(n_samples=1)
        one_base_point = self.space.random_uniform(n_samples=1)
        n_tangent_vecs = self.space.random_uniform(n_samples=n_samples)
        n_base_points = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.exp(one_tangent_vec, one_base_point)
        expected = one_tangent_vec + one_base_point
        self.assertTrue(np.allclose(result, expected))

        result = self.metric.exp(n_tangent_vecs, one_base_point)
        self.assertTrue(np.allclose(result.shape, (n_samples, dim)),
                        '\n result.shape = {}'.format(result.shape))

        result = self.metric.exp(one_tangent_vec, n_base_points)
        self.assertTrue(np.allclose(result.shape, (n_samples, dim)))

        result = self.metric.exp(n_tangent_vecs, n_base_points)
        self.assertTrue(np.allclose(result.shape, (n_samples, dim)))

    def test_log(self):
        base_point = np.array([0, 1])
        point = np.array([2, 10])

        result = self.metric.log(point=point, base_point=base_point)
        expected = point - base_point
        self.assertTrue(np.allclose(result, expected))

    def test_log_vectorization(self):
        n_samples = self.n_samples
        dim = self.dimension
        one_point = self.space.random_uniform(n_samples=1)
        one_base_point = self.space.random_uniform(n_samples=1)
        n_points = self.space.random_uniform(n_samples=n_samples)
        n_base_points = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.log(one_point, one_base_point)
        expected = one_point - one_base_point
        self.assertTrue(np.allclose(result, expected))

        result = self.metric.log(n_points, one_base_point)
        self.assertTrue(np.allclose(result.shape, (n_samples, dim)))

        result = self.metric.log(one_point, n_base_points)
        self.assertTrue(np.allclose(result.shape, (n_samples, dim)))

        result = self.metric.log(n_points, n_base_points)
        self.assertTrue(np.allclose(result.shape, (n_samples, dim)))

    def test_squared_dist(self):
        point_a = np.array([-1, 4])
        point_b = np.array([1, 1])

        result = self.metric.squared_dist(point_a, point_b)
        vec = point_b - point_a
        expected = np.dot(vec, vec)
        self.assertTrue(np.allclose(result, expected))

    def test_squared_dist_vectorization(self):
        n_samples = self.n_samples
        one_point_a = self.space.random_uniform(n_samples=1)
        one_point_b = self.space.random_uniform(n_samples=1)
        n_points_a = self.space.random_uniform(n_samples=n_samples)
        n_points_b = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.squared_dist(one_point_a, one_point_b)
        vec = one_point_a - one_point_b
        expected = np.dot(vec, vec.transpose())
        self.assertTrue(np.allclose(result, expected))

        result = self.metric.squared_dist(n_points_a, one_point_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

        result = self.metric.squared_dist(one_point_a, n_points_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

        result = self.metric.squared_dist(n_points_a, n_points_b)
        expected = np.zeros((n_samples, 1))
        for i in range(n_samples):
            vec = n_points_a[i] - n_points_b[i]
            expected[i] = np.dot(vec, vec.transpose())
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))
        self.assertTrue(np.allclose(result, expected))

    def test_dist(self):
        point_a = np.array([0, 1])
        point_b = np.array([2, 10])

        result = self.metric.dist(point_a, point_b)
        expected = np.linalg.norm(point_b - point_a)
        self.assertTrue(np.allclose(result, expected))

    def test_dist_vectorization(self):
        n_samples = self.n_samples
        one_point_a = self.space.random_uniform(n_samples=1)
        one_point_b = self.space.random_uniform(n_samples=1)
        n_points_a = self.space.random_uniform(n_samples=n_samples)
        n_points_b = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.dist(one_point_a, one_point_b)
        vec = one_point_a - one_point_b
        expected = np.sqrt(np.dot(vec, vec.transpose()))
        self.assertTrue(np.allclose(result, expected))

        result = self.metric.dist(n_points_a, one_point_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

        result = self.metric.dist(one_point_a, n_points_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

        result = self.metric.dist(n_points_a, n_points_b)
        expected = np.zeros((n_samples, 1))
        for i in range(n_samples):
            vec = n_points_a[i] - n_points_b[i]
            expected[i] = np.sqrt(np.dot(vec, vec.transpose()))
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))
        self.assertTrue(np.allclose(result, expected))

    def test_random_uniform_and_belongs(self):
        point = self.space.random_uniform()
        self.assertTrue(self.space.belongs(point))

    def test_random_uniform_and_belongs_vectorization(self):
        n_samples = self.n_samples
        n_points = self.space.random_uniform(n_samples=n_samples)
        self.assertTrue(np.all(self.space.belongs(n_points)))

    def test_geodesic_and_belongs(self):
        initial_point = self.space.random_uniform()
        initial_tangent_vec = np.array([2., 0.])
        geodesic = self.metric.geodesic(
                                   initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)

        t = np.linspace(start=0, stop=1, num=100)
        points = geodesic(t)
        self.assertTrue(np.all(self.space.belongs(points)))

    def test_mean(self):
        point = np.array([1, 4])
        result = self.metric.mean(points=[point, point, point])
        expected = point

        self.assertTrue(np.allclose(result, expected))

        points = np.array([[1, 2],
                           [2, 3],
                           [3, 4],
                           [4, 5]])
        weights = np.array([1, 2, 1, 2])

        result = self.metric.mean(points, weights)
        expected = np.array([16., 22.]) / 6.
        self.assertTrue(np.allclose(result, expected))

    def test_variance(self):
        points = np.array([[1, 2],
                           [2, 3],
                           [3, 4],
                           [4, 5]])
        weights = np.array([1, 2, 1, 2])
        base_point = np.zeros(2)
        result = self.metric.variance(points, weights, base_point)
        # we expect the average of the points' sq norms.
        expected = (1 * 5. + 2 * 13. + 1 * 25. + 2 * 41.) / 6.
        self.assertTrue(np.allclose(result, expected))

if __name__ == '__main__':
        unittest.main()
