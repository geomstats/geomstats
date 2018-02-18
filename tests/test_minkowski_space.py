"""Unit tests for minkowski space module."""

import numpy as np
import unittest

from geomstats.minkowski_space import MinkowskiSpace


class TestMinkowskiSpaceMethods(unittest.TestCase):
    def setUp(self):
        self.time_like_dim = 0
        self.dimension = 2
        self.space = MinkowskiSpace(self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10

    def test_inner_product_matrix(self):
        result = self.metric.inner_product_matrix()

        expected = np.eye(self.dimension)
        expected[self.time_like_dim, self.time_like_dim] = -1
        self.assertTrue(np.allclose(result, expected))

    def test_inner_product(self):
        point_a = np.array([0, 1])
        point_b = np.array([2, 10])

        result = self.metric.inner_product(point_a, point_b)
        expected = np.dot(point_a, point_b)
        expected -= (2 * point_a[self.time_like_dim]
                     * point_b[self.time_like_dim])
        self.assertTrue(np.allclose(result, expected))

    def test_inner_product_vectorization(self):
        n_samples = self.n_samples
        one_point_a = self.space.random_uniform(n_samples=1)
        one_point_b = self.space.random_uniform(n_samples=1)
        n_points_a = self.space.random_uniform(n_samples=n_samples)
        n_points_b = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.inner_product(one_point_a, one_point_b)
        expected = np.dot(one_point_a, one_point_b.transpose())
        expected -= (2 * one_point_a[:, self.time_like_dim]
                     * one_point_b[:, self.time_like_dim])
        self.assertTrue(np.allclose(result, expected))

        result = self.metric.inner_product(n_points_a, one_point_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

        result = self.metric.inner_product(one_point_a, n_points_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

        result = self.metric.inner_product(n_points_a, n_points_b)
        expected = np.zeros((n_samples, 1))
        for i in range(n_samples):
            expected[i] = np.dot(n_points_a[i], n_points_b[i])
            expected[i] -= (2 * n_points_a[i, self.time_like_dim]
                            * n_points_b[i, self.time_like_dim])
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))
        self.assertTrue(np.allclose(result, expected))

    def test_squared_norm(self):
        point = np.array([-2, 4])

        result = self.metric.squared_norm(point)
        expected = np.dot(point, point)
        expected -= 2 * point[self.time_like_dim] * point[self.time_like_dim]
        self.assertTrue(np.allclose(result, expected))

    def test_squared_norm_vectorization(self):
        n_samples = self.n_samples
        n_points = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.squared_norm(n_points)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

    def test_norm(self):
        point = np.array([-1, 4])
        self.assertRaises(ValueError,
                          lambda: self.metric.norm(point))

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
        expected -= 2 * vec[self.time_like_dim] * vec[self.time_like_dim]
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
        expected -= 2 * vec[0, self.time_like_dim] * vec[0, self.time_like_dim]
        self.assertTrue(np.allclose(result, expected))

        result = self.metric.squared_dist(n_points_a, one_point_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

        result = self.metric.squared_dist(one_point_a, n_points_b)
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))

        result = self.metric.squared_dist(n_points_a, n_points_b)
        expected = np.zeros((n_samples, 1))
        for i in range(n_samples):
            vec = n_points_a[i] - n_points_b[i]
            expected_i = np.dot(vec, vec.transpose())
            expected_i -= 2 * vec[self.time_like_dim] * vec[self.time_like_dim]
            expected[i] = expected_i
        self.assertTrue(np.allclose(result.shape, (n_samples, 1)))
        self.assertTrue(np.allclose(result, expected))

    def test_dist(self):
        point_a = np.array([-1, 4])
        point_b = np.array([1, 1])
        self.assertRaises(ValueError,
                          lambda: self.metric.dist(point_a, point_b))

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
        # we expect the average of the points' Minkowski sq norms.
        expected = (1 * 3. + 2 * 5. + 1 * 7. + 2 * 9.) / 6.
        self.assertTrue(np.allclose(result, expected))


if __name__ == '__main__':
        unittest.main()
