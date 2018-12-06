"""
Unit tests for Lie groups.
"""

import geomstats.backend as gs
import geomstats.tests

from geomstats.lie_group import LieGroup


class TestLieGroupMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        self.dimension = 4
        self.group = LieGroup(self.dimension)

    def test_dimension(self):
        result = self.group.dimension
        expected = self.dimension

        with self.session():
            self.assertAllClose(result, expected)


if __name__ == '__main__':
        geomstats.tests.main()
