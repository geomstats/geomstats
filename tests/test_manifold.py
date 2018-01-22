"""Unit tests for base_manifolds module."""

import numpy as np
import unittest

from geomstats.manifold import Manifold


class TestManifoldMethods(unittest.TestCase):
    DIMENSION = np.random.randint(1)
    MANIFOLD = Manifold(DIMENSION)

    def test_dimension(self):
        result = self.MANIFOLD.dimension
        expected = self.DIMENSION
        self.assertTrue(np.allclose(result, expected))

    def test_belongs(self):
        point = np.array([1, 2, 3])
        self.assertRaises(NotImplementedError,
                          lambda: self.MANIFOLD.belongs(point))

    def test_regularize(self):
        point = np.array(5)

        result = self.MANIFOLD.regularize(point)
        expected = point
        self.assertTrue(np.allclose(result, expected))


if __name__ == '__main__':
        unittest.main()
