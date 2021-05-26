"""Unit tests for Lie groups."""

import geomstats.tests
from geomstats.geometry.lie_group import LieGroup


class TestLieGroup(geomstats.tests.TestCase):

    def test_dimension(self):
        self.assertRaises(TypeError, lambda: LieGroup(4))
