"""Unit tests for base_manifolds module."""

import numpy as np
import unittest

from geomstats.manifold import Manifold


class TestManifoldMethods(unittest.TestCase):
    DIMENSION = np.random.randint(1)

    def test_dimension(self):
        manifold = Manifold(self.DIMENSION)

        result = manifold.dimension
        expected = self.DIMENSION
        self.assertTrue(np.allclose(result, expected))

    def test_regularize(self):
        manifold = Manifold(self.DIMENSION)
        point = np.array(5)

        result = manifold.regularize(point)
        expected = point
        self.assertTrue(np.allclose(result, expected))


if __name__ == '__main__':
        unittest.main()
