"""
Unit tests for manifolds for tensorflow backend.
"""

import importlib
import os
import tensorflow as tf

import geomstats.backend as gs

from geomstats.manifold import Manifold


class TestManifoldMethodsTensorFlow(tf.test.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        self.dimension = 4
        self.manifold = Manifold(self.dimension)

    @classmethod
    def setUpClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        importlib.reload(gs)

    @classmethod
    def tearDownClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'numpy'
        importlib.reload(gs)

    def test_dimension(self):
        result = self.manifold.dimension
        expected = self.dimension
        with self.test_session():
            self.assertAllClose(result, expected)

    def test_belongs(self):
        point = gs.array([1, 2, 3])
        self.assertRaises(NotImplementedError,
                          lambda: self.manifold.belongs(point))

    def test_regularize(self):
        point = gs.array([1, 2, 3])
        result = self.manifold.regularize(point)
        expected = point
        with self.test_session():
            self.assertAllClose(result, expected)


if __name__ == '__main__':
        tf.test.main()
