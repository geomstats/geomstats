"""
Unit tests for Lie groups.
"""

import unittest

import geomstats.backend as gs

from geomstats.lie_group import LieGroup


class TestLieGroupMethods(unittest.TestCase):
    _multiprocess_can_split_ = True

    dimension = 4
    group = LieGroup(dimension=dimension,
                     identity=gs.zeros(4))


if __name__ == '__main__':
        unittest.main()
