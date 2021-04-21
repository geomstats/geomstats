"""Unit tests for Lie groups."""

import geomstats.tests
from geomstats.geometry.lie_group import LieGroup

ATOL = 1e-6


class TestLieGroup(geomstats.tests.TestCase):
    dimension = 4
    group = LieGroup(dim=dimension)

    def test_dimension(self):
        result = self.group.dim
        expected = self.dimension

        self.assertAllClose(result, expected)
