"""Unit tests for base_manifolds module."""

import numpy as np
import unittest

from geomstats.manifold import Manifold


class TestManifoldMethods(unittest.TestCase):
    def setUp(self):
        self.dimension = np.random.randint(1)
        self.manifold = Manifold(self.dimension)

    def test_dimension(self):
        result = self.manifold.dimension
        expected = self.dimension
        self.assertTrue(np.allclose(result, expected))

    def test_belongs(self):
        point = np.array([1, 2, 3])
        self.assertRaises(NotImplementedError,
                          lambda: self.manifold.belongs(point))

    def test_regularize(self):
        point = np.array(5)

        result = self.manifold.regularize(point)
        expected = point
        self.assertTrue(np.allclose(result, expected))


if __name__ == '__main__':
        unittest.main()
