"""Unit tests for hypersphere module."""

import os
os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
from geomstats import hypersphere


import geomstats.backend as gs
import importlib

import tensorflow as tf



class TestHypersphereMethods(tf.test.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        self.dimension = 4
        self.space = hypersphere.Hypersphere(dimension=self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10
        self.depth = 3

    @classmethod
    def setUpClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
        importlib.reload(gs)
        importlib.reload(hypersphere)

    @classmethod
    def tearDownClass(cls):
        os.unsetenv('GEOMSTATS_BACKEND')
        importlib.reload(gs)

    def test_random_uniform_and_belongs(self):
        """
        Test that the random uniform method samples
        on the hypersphere space.
        """
        point = self.space.random_uniform()
        with self.test_session():
            self.assertTrue(self.space.belongs(point).eval())


if __name__ == '__main__':
    tf.test.main()
