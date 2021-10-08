"""Testing class for geomstats.

This class abstracts the backend type.
"""

import os
import unittest

import numpy as np

import geomstats.backend as gs


def autograd_backend():
    """Check if autograd is set as backend."""
    return os.environ["GEOMSTATS_BACKEND"] == "autograd"


def np_backend():
    """Check if numpy is set as backend."""
    return os.environ["GEOMSTATS_BACKEND"] == "numpy"


def pytorch_backend():
    """Check if pytorch is set as backend."""
    return os.environ["GEOMSTATS_BACKEND"] == "pytorch"


def tf_backend():
    """Check if tensorflow is set as backend."""
    return os.environ["GEOMSTATS_BACKEND"] == "tensorflow"


def autograd_only(test_item):
    """Decorate to filter tests for autograd only."""
    if autograd_backend():
        return test_item
    return unittest.skip("Test for autograd backend only.")(test_item)


def np_only(test_item):
    """Decorate to filter tests for numpy only."""
    if np_backend():
        return test_item
    return unittest.skip("Test for numpy backend only.")(test_item)


def np_and_tf_only(test_item):
    """Decorate to filter tests for numpy and tensorflow only."""
    if np_backend() or tf_backend():
        return test_item
    return unittest.skip("Test for numpy and tensorflow backends only.")(test_item)


def np_and_torch_only(test_item):
    """Decorate to filter tests for numpy and pytorch only."""
    if np_backend() or pytorch_backend():
        return test_item
    return unittest.skip("Test for numpy and pytorch backends only.")(test_item)


def np_and_autograd_only(test_item):
    """Decorate to filter tests for numpy and autograd only."""
    if np_backend() or autograd_backend():
        return test_item
    return unittest.skip("Test for numpy and autograd backends only.")(test_item)


def autograd_and_torch_only(test_item):
    """Decorate to filter tests for autograd and torch only."""
    if autograd_backend() or pytorch_backend():
        return test_item
    return unittest.skip("Test for autograd and torch backends only.")(test_item)


def torch_only(test_item):
    """Decorate to filter tests for pytorch only."""
    if pytorch_backend():
        return test_item
    return unittest.skip("Test for pytorch backend only.")(test_item)


def tf_only(test_item):
    """Decorate to filter tests for tensorflow only."""
    if tf_backend():
        return test_item
    return unittest.skip("Test for tensorflow backend only.")(test_item)


def np_autograd_and_tf_only(test_item):
    """Decorate to filter tests for numpy, autograd and tf only."""
    if np_backend() or autograd_backend() or tf_backend():
        return test_item
    return unittest.skip("Test for numpy, autograd and tensorflow backends only.")(
        test_item
    )


def autograd_and_tf_only(test_item):
    """Decorate to filter tests for autograd and tensorflow only."""
    if autograd_backend() or tf_backend():
        return test_item
    return unittest.skip("Test for autograd and tensorflow backends only.")(test_item)


def np_autograd_and_torch_only(test_item):
    """Decorate to filter tests for numpy, autograd and torch only."""
    if np_backend() or autograd_backend() or pytorch_backend():
        return test_item
    return unittest.skip("Test for numpy, autograd and pytorch backends only.")(
        test_item
    )


def autograd_tf_and_torch_only(test_item):
    """Decorate to filter tests for backends with autodiff only."""
    if np_backend():
        return unittest.skip("Test for backends with automatic differentiation only.")(
            test_item
        )
    return test_item


_TestBaseClass = unittest.TestCase
if tf_backend():
    import tensorflow as tf

    _TestBaseClass = tf.test.TestCase

if pytorch_backend():
    import torch


def pytorch_error_msg(a, b, rtol, atol):
    msg = f"\ntensor 1\n{a}\ntensor 2\n{b}"
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.dtype == torch.bool and b.dtype == torch.bool:
            diff = torch.logical_xor(a, b)
            msg = msg + f"\ndifference \n{diff}"
        else:
            diff = torch.abs(a - b)
            msg = msg + f"\ndifference \n{diff}\nrtol {rtol}\natol {atol}"
    return msg


class TestCase(_TestBaseClass):
    def assertAllClose(self, a, b, rtol=gs.rtol, atol=gs.atol):
        if tf_backend():
            return super().assertAllClose(a, b, rtol=rtol, atol=atol)
        if np_backend() or autograd_backend():
            return np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

        return self.assertTrue(
            gs.allclose(a, b, rtol=rtol, atol=atol),
            msg=pytorch_error_msg(a, b, rtol, atol),
        )

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
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
