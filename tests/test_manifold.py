"""Unit tests for manifolds."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.manifold import Manifold


class TestManifold(geomstats.tests.TestCase):
    def setUp(self):
        self.dimension = 4
        self.manifold = Manifold(self.dimension)

    def test_dimension(self):
        result = self.manifold.dim
        expected = self.dimension
        self.assertAllClose(result, expected)

    def test_belongs(self):
        point = gs.array([1.0, 2.0, 3.0])
        self.assertRaises(NotImplementedError, lambda: self.manifold.belongs(point))

    def test_regularize(self):
        point = gs.array([1.0, 2.0, 3.0])
        result = self.manifold.regularize(point)
        expected = point
        self.assertAllClose(result, expected)
