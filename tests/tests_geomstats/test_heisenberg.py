"""Unit tests for the 3D heisenberg group in vector representation."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.heisenberg import HeisenbergVectors


class TestHeisenbergVectors(geomstats.tests.TestCase):
    def setup_method(self):
        self.dimension = 3
        self.group = HeisenbergVectors()

    def test_dimension(self):
        result = self.group.dim
        expected = self.dimension
        self.assertAllClose(result, expected)

    def test_belongs(self):
        point = gs.array([1.0, 2.0, 3.0, 4])
        result = self.group.belongs(point)
        expected = False

        self.assertAllClose(result, expected)

    def test_belongs_vectorization(self):
        point = gs.array([[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]])
        result = self.group.belongs(point)
        expected = gs.array([False, False])

        self.assertAllClose(result, expected)

    def test_is_tangent(self):
        vector = gs.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        result = self.group.is_tangent(vector)
        expected = gs.array([False, False])

        self.assertAllClose(result, expected)

    def test_jacobian_translation(self):
        vector = gs.array([[1.0, -10.0, 0.2], [-2.0, 100.0, 0.5]])
        result = self.group.jacobian_translation(vector)
        expected = gs.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 0.5, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-50.0, -1.0, 1.0]],
            ]
        )

        self.assertAllClose(result, expected)

    def test_random_point_belongs(self):
        n_samples = 2
        bound = 1
        points = self.group.random_point(n_samples=n_samples, bound=bound)
        result = self.group.belongs(points)
        expected = gs.array([True, True])

        self.assertAllClose(result, expected)
