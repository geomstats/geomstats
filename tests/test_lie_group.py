"""Unit tests for Lie groups."""

import geomstats.tests
from geomstats.geometry.lie_group import LieGroup

ATOL = 1e-6

class DummyLieGroup(LieGroup):
    def belongs(self, point, atol=ATOL):
        pass

class TestLieGroup(geomstats.tests.TestCase):
    dimension = 4
    group = DummyLieGroup(dim=dimension)

    def test_dimension(self):
        result = self.group.dim
        expected = self.dimension

        self.assertAllClose(result, expected)
