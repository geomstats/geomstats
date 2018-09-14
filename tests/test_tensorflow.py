"""
Parent class for unit tests on tensorflow.
"""

import io
import tensorflow as tf
import unittest

from tests.test_backend_tensorflow import TestBackendTensorFlow
from tests.test_hypersphere_tensorflow import TestHypersphereOnTensorFlow


if __name__ == '__main__':
    tf.enable_eager_execution()
    stream = io.StringIO()

    runner = unittest.TextTestRunner(stream=stream)
    result = runner.run(unittest.makeSuite(TestBackendTensorFlow))
    #stream.seek(0)
    #print(stream.read())

    result = runner.run(unittest.makeSuite(TestHypersphereOnTensorFlow))
    stream.seek(0)
    print(stream.read())
