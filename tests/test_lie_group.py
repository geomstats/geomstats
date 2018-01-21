"""Unit tests for lie group module."""

import numpy as np
import unittest

from geomstats.lie_group import LieGroup


class TestLieGroupMethods(unittest.TestCase):
    DIMENSION = 4
    GROUP = LieGroup(dimension=DIMENSION,
                     identity=np.zeros(4))

if __name__ == '__main__':
        unittest.main()
