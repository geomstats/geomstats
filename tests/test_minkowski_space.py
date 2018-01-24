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

    def test_squared_norm(self):
        point = np.array([-2, 4])

        result = self.metric.squared_norm(point)
        expected = np.dot(point, point)
        expected -= 2 * point[self.time_like_dim] * point[self.time_like_dim]
        self.assertTrue(np.allclose(result, expected))

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

    def test_log(self):
        base_point = np.array([0, 1])
        point = np.array([2, 10])

        result = self.metric.log(point=point, base_point=base_point)
        expected = point - base_point
        self.assertTrue(np.allclose(result, expected))

    def test_squared_dist(self):
        point_a = np.array([-1, 4])
        point_b = np.array([1, 1])

        result = self.metric.squared_dist(point_a, point_b)
        vec = point_b - point_a
        expected = np.dot(vec, vec)
        expected -= 2 * vec[self.time_like_dim] * vec[self.time_like_dim]
        self.assertTrue(np.allclose(result, expected))

    def test_dist(self):
        point_a = np.array([-1, 4])
        point_b = np.array([1, 1])
        self.assertRaises(ValueError,
                          lambda: self.metric.dist(point_a, point_b))

    def test_random_uniform_and_belongs(self):
        point = self.space.random_uniform()
        self.assertTrue(self.space.belongs(point))

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
