"""Unit tests for lie group module."""

import numpy as np
import unittest

from geomstats.lie_group import LieGroup


class TestLieGroupMethods(unittest.TestCase):
    dimension = 4
    group = LieGroup(dimension=dimension,
                     identity=np.zeros(4))

if __name__ == '__main__':
        unittest.main()
