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
        self.dimension = gs.random.randint(low=1, high=10)
        self.manifold = Manifold(self.dimension)

    @classmethod
    def setUpClass(cls):
        tf.enable_eager_execution()
        os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
        importlib.reload(gs)

    @classmethod
    def tearDownClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = 'numpy'
        importlib.reload(gs)

    def test_dimension(self):
        result = self.manifold.dimension
        expected = self.dimension
        with self.test_session():
            self.assertAllClose(gs.eval(result), gs.eval(expected))

    def test_belongs(self):
        point = gs.array([1, 2, 3])
        self.assertRaises(NotImplementedError,
                          lambda: self.manifold.belongs(point))

    def test_regularize(self):
        point = gs.array([1, 2, 3])
        self.assertRaises(NotImplementedError,
                          lambda: self.manifold.regularize(point))


if __name__ == '__main__':
        tf.test.main()
