"""
Unit tests for Lie groups.
"""

import geomstats.tests
from geomstats.geometry.lie_group import LieGroup


class TestLieGroupMethods(geomstats.tests.TestCase):
    dimension = 4
    group = LieGroup(dimension=dimension)

    def test_dimension(self):
        result = self.group.dimension
        expected = self.dimension

        self.assertAllClose(result, expected)
