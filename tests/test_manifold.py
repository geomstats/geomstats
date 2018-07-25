"""
Unit tests for manifolds.
"""

import unittest

import geomstats.backend as gs

from geomstats.manifold import Manifold


class TestManifoldMethods(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        self.dimension = gs.random.randint(low=1, high=10)
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
        self.assertRaises(NotImplementedError,
                          lambda: self.manifold.regularize(point))


if __name__ == '__main__':
        unittest.main()
