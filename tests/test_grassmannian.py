"""
Unit tests for the Grassmannian.
"""

import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry.grassmannian import Grassmannian


class TestGrassmannianMethods(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.n = 4
        self.p = 2
        self.space = Grassmannian(self.n, self.p)
