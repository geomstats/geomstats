"""
Testing class for geomstats.

This class abstracts the backend type.
"""

import os
import tensorflow as tf
import unittest

import geomstats.backend as gs


def pytorch_backend():
    return os.environ['GEOMSTATS_BACKEND'] == 'pytorch'


def tf_backend():
    return os.environ['GEOMSTATS_BACKEND'] == 'tensorflow'


def np_backend():
    return os.environ['GEOMSTATS_BACKEND'] == 'numpy'


test_class = unittest.TestCase
if tf_backend():
    test_class = tf.test.TestCase


def np_only(test_item):
    """Decorator to filter tests for numpy only."""
    if not np_backend():
        test_item.__unittest_skip__ = True
        test_item.__unittest_skip_why__ = (
            'Test for numpy backend only.')
    return test_item


def np_and_tf_only(test_item):
    """Decorator to filter tests for numpy and tensorflow only."""
    if not (np_backend() or tf_backend()):
        test_item.__unittest_skip__ = True
        test_item.__unittest_skip_why__ = (
            'Test for numpy and tensorflow backends only.')
    return test_item


class DummySession():
    def __enter__(self):
        pass

    def __exit__(self, a, b, c):
        pass


class TestCase(test_class):

    def assertAllClose(self, a, b, rtol=1e-6, atol=1e-6):
        if tf_backend():
            return super().assertAllClose(a, b, rtol=rtol, atol=atol)
        return self.assertTrue(gs.allclose(a, b, rtol=rtol, atol=atol))

    def session(self):
        if tf_backend():
            return super().test_session()
        return DummySession()

    def assertShapeEqual(self, a, b):
        if tf_backend():
            return super().assertShapeEqual(a, b)
        super().assertEqual(a.shape, b.shape)

    @classmethod
    def setUpClass(cls):
        if tf_backend():
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
