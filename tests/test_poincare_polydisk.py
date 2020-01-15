"""
Unit tests for the Poincare Polydisk.
"""


import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry.poincare_polydisk import PoincarePolydisk


class TestPoincarePolydiskMethods(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)
        self.n_disks = 5
        self.space = PoincarePolydisk(n_disks=self.n_disks)

    def test_dimension(self):
        expected = self.n_disks * 2
        result = self.space.dimension
        self.assertAllClose(result, expected)