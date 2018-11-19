"""
Unit tests for the affine connections for tensorflow backend.
"""

import importlib
import os
import tensorflow as tf

import geomstats.backend as gs

from geomstats.connection import LeviCivitaConnection
from geomstats.euclidean_space import EuclideanMetric


class TestConnectionMethods(tf.test.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.dimension = 4
        self.metric = EuclideanMetric(dimension=self.dimension)
        self.connection = LeviCivitaConnection(self.metric)

    @classmethod
    def setUpClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
        importlib.reload(gs)

    @classmethod
    def tearDownClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'numpy'
        importlib.reload(gs)

    def test_metric_matrix(self):
        base_point = tf.convert_to_tensor([0., 1., 0., 0.])

        result = self.connection.metric_matrix(base_point)
        expected = tf.convert_to_tensor([gs.eye(self.dimension)])

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_cometric_matrix(self):
        base_point = tf.convert_to_tensor([0., 1., 0., 0.])

        result = self.connection.cometric_matrix(base_point)
        expected = tf.convert_to_tensor([gs.eye(self.dimension)])

        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))


if __name__ == '__main__':
        tf.test.main()
