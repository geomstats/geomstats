"""
Unit tests for Lie groups for tensorflow backend.
"""

import importlib
import os
import tensorflow as tf

import geomstats.backend as gs

from geomstats.lie_group import LieGroup


class TestLieGroupMethods(tf.test.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        self.dimension = 4
        self.group = LieGroup(self.dimension)

    @classmethod
    def setUpClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
        importlib.reload(gs)

    @classmethod
    def tearDownClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'numpy'
        importlib.reload(gs)

    def test_dimension(self):
        result = self.group.dimension
        expected = self.dimension

        with self.test_session():
            self.assertAllClose(result, expected)


if __name__ == '__main__':
        tf.test.main()
