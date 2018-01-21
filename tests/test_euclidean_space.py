"""Unit tests for euclidean space module."""

import numpy as np
import unittest

from geomstats.euclidean_space import EuclideanSpace


class TestEuclideanSpaceMethods(unittest.TestCase):
    DIMENSION = 2
    SPACE = EuclideanSpace(DIMENSION)
    METRIC = SPACE.metric

    def test_inner_product_matrix(self):
        result = self.METRIC.inner_product_matrix()

        expected = np.eye(self.DIMENSION)
        self.assertTrue(np.allclose(result, expected))

    def test_inner_product(self):
        point_a = np.array([0, 1])
        point_b = np.array([2, 10])

        result = self.METRIC.inner_product(point_a, point_b)
        expected = np.dot(point_a, point_b)
        self.assertTrue(np.allclose(result, expected))

    def test_squared_norm(self):
        point = np.array([-2, 4])

        result = self.METRIC.squared_norm(point)
        expected = np.linalg.norm(point) ** 2
        self.assertTrue(np.allclose(result, expected))

    def test_norm(self):
        point = np.array([-2, 4])

        result = self.METRIC.norm(point)
        expected = np.linalg.norm(point)
        self.assertTrue(np.allclose(result, expected))

    def test_exp(self):
        base_point = np.array([0, 1])
        vector = np.array([2, 10])

        result = self.METRIC.exp(tangent_vec=vector,
                                 base_point=base_point)
        expected = base_point + vector
        self.assertTrue(np.allclose(result, expected))

    def test_log(self):
        base_point = np.array([0, 1])
        point = np.array([2, 10])

        result = self.METRIC.log(point=point, base_point=base_point)
        expected = point - base_point
        self.assertTrue(np.allclose(result, expected))

    def test_dist(self):
        point_a = np.array([0, 1])
        point_b = np.array([2, 10])

        result = self.METRIC.dist(point_a, point_b)
        expected = np.linalg.norm(point_b - point_a)
        self.assertTrue(np.allclose(result, expected))

    def test_random_uniform_and_belongs(self):
        point = self.METRIC.random_uniform()
        self.assertTrue(self.SPACE.belongs(point))

    def test_mean(self):
        points = np.array([[1, 2],
                           [2, 3],
                           [3, 4],
                           [4, 5]])
        weights = np.array([1, 2, 1, 2])

        result = self.METRIC.mean(points, weights)
        expected = np.array([16., 22.]) / 6.
        self.assertTrue(np.allclose(result, expected))

    def test_variance(self):
        points = np.array([[1, 2],
                           [2, 3],
                           [3, 4],
                           [4, 5]])
        weights = np.array([1, 2, 1, 2])
        base_point = np.zeros(2)
        result = self.METRIC.variance(points, weights, base_point)
        # we expect the average of the points' sq norms.
        expected = (1 * 5. + 2 * 13. + 1 * 25. + 2 * 41.) / 6.
        self.assertTrue(np.allclose(result, expected))


if __name__ == '__main__':
        unittest.main()
