"""Unit tests for base_manifolds module."""

from geomstats.manifold import Manifold

import geomstats.backend as gs
import unittest


class TestManifoldMethods(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        self.dimension = 10
        self.manifold = Manifold(self.dimension)

    def test_dimension(self):
        result = self.manifold.dimension
        expected = self.dimension
        self.assertTrue(gs.allclose(result, expected))

    def test_belongs(self):
        point = gs.array([1, 2, 3])
        self.assertRaises(NotImplementedError,
                          lambda: self.manifold.belongs(point))

    def test_regularize(self):
        point = gs.array([1, 2, 3])
        self.assertTrue(gs.allclose(point, self.manifold.regularize(point)))

    def test_get_dimension(self):
        manifold = Manifold(10)
        self.assertEquals(manifold.get_dimension(), 10)


if __name__ == '__main__':
        unittest.main()
