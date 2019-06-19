"""
Unit tests for tensorflow backend.
"""

import importlib
import os
import tensorflow as tf

import geomstats.backend as gs


class TestBackendTensorFlow(tf.test.TestCase):
    _multiprocess_can_split_ = True

    @classmethod
    def setUpClass(cls):
        cls.initial_backend = os.environ['GEOMSTATS_BACKEND']
        os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        importlib.reload(gs)

    @classmethod
    def tearDownClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = cls.initial_backend
        importlib.reload(gs)

    def test_vstack(self):
        with self.test_session():
            tensor_1 = tf.convert_to_tensor([[1., 2., 3.], [4., 5., 6.]])
            tensor_2 = tf.convert_to_tensor([[7., 8., 9.]])

            result = gs.vstack([tensor_1, tensor_2])
            expected = tf.convert_to_tensor([
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.]])
            self.assertAllClose(result, expected)

    def test_tensor_addition(self):
        with self.test_session():
            tensor_1 = gs.ones((1, 1))
            tensor_2 = gs.ones((0, 1))

            result = tensor_1 + tensor_2


if __name__ == '__main__':
    tf.test.main()
