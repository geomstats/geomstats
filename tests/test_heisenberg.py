"""Unit tests for the 3D heisenberg group in vector representation."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.heisenberg import heisenbergVectors


class TestHeisenbergVectors(geomstats.tests.TestCase):

    def setUp(self):
        self.dimension = 3
        self.group = heisenbergVectors()

    def test_dimension(self):
        result = self.group.dim
        expected = self.dimension
        self.assertAllClose(result, expected)

    def test_belongs(self):
        point = gs.array([1., 2., 3., 4])
        result = self.group.belongs(point)
        expected = False

        self.assertAllClose(result, expected)

    def test_belongs_vectorization(self):
        point = gs.array([
            [1., 2., 3., 1.], [4., 5., 6., 1.]])
        result = self.group.belongs(point)
        expected = gs.array([False, False])

        self.assertAllClose(result, expected)

    def test_is_tangent(self):
        vector = gs.array([1., 2., 3., 4.])
        result = self.group.is_tangent(vector)
        expected = False

        self.assertAllClose(result, expected)

    def test_random_point_belongs(self):
        n_samples = 2
        bound = 1
        points = self.group.random_point(n_samples=n_samples, bound=bound)
        result = self.group.belongs(points)
        expected = gs.array([True, True])

        self.assertAllClose(result, expected)
