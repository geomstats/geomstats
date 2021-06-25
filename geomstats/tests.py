"""Testing class for geomstats.

This class abstracts the backend type.
"""

import os
import numpy as np
import pytest

import geomstats.backend as gs


def pytorch_backend():
    """Check if pytorch is set as backend."""
    return os.environ['GEOMSTATS_BACKEND'] == 'pytorch'


def tf_backend():
    """Check if tensorflow is set as backend."""
    return os.environ['GEOMSTATS_BACKEND'] == 'tensorflow'


def np_backend():
    """Check if numpy is set as backend."""
    return os.environ['GEOMSTATS_BACKEND'] == 'numpy'



pytorch_only = pytest.mark.skipif(not np_backend()) 
np_only = pytest.mark.skipif(not np_backend())
tf_only = pytest.mark.skipif(not tf_backend())
np_and_tf_only = pytest.mark.skipif(pytorch_backend())
np_and_pytorch_only = pytest.mark.skipif(tf_backend())

_TestBaseClass = object 
if tf_backend():
    import tensorflow as tf
    _TestBaseClass = tf.test.TestCase


class TestCase(_TestBaseClass):
    def assertAllClose(self, a, b, rtol=gs.rtol, atol=gs.atol):
        if tf_backend():
            return super().assertAllClose(a, b, rtol=rtol, atol=atol)
        if np_backend():
            return np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        assert gs.allclose(a, b, rtol=rtol, atol=atol)

    def assertAllCloseToNp(self, a, np_a, rtol=gs.rtol, atol=gs.atol):
        are_same_shape = np.all(a.shape == np_a.shape)
        are_same = np.allclose(a, np_a, rtol=rtol, atol=atol)
        if tf_backend():
            return super().assertTrue(are_same_shape and are_same)
        return super().assertTrue(are_same_shape and are_same)

    def assertShapeEqual(self, a, b):
        if tf_backend():
            return super().assertShapeEqual(a, b)
        return super().assertEqual(a.shape, b.shape)

    @classmethod
    def setUpClass(cls):
        if tf_backend():
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
