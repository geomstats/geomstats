"""
Unit tests for Lie groups.
"""

import geomstats.tests

from geomstats.lie_group import LieGroup


class TestLieGroupMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    dimension = 4
    group = LieGroup(dimension=dimension)

    def test_dimension(self):
        result = self.group.dimension
        expected = self.dimension

        self.assertAllClose(result, expected)


if __name__ == '__main__':
        geomstats.tests.main()
