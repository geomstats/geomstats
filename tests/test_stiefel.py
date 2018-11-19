"""
Unit tests for Stiefel manifolds.
"""

import unittest

import geomstats.backend as gs

from geomstats.stiefel import Stiefel


class TestStiefelMethods(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)

        self.p = 6
        self.n = 10
        self.space = Stiefel(self.p, self.n)
        self.n_samples = 10

    def test_belongs(self):
        point = self.space.random_uniform()
        belongs = self.space.belongs(point)

        gs.testing.assert_allclose(belongs.shape, (1, 1))

    def test_random_uniform(self):
        point = self.space.random_uniform()

        gs.testing.assert_allclose(point.shape, (1, self.dimension + 1))
