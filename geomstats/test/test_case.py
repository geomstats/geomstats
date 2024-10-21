import inspect

import numpy as np
import pytest

import geomstats.backend as gs

# TODO: remove backend influence
# TODO: simplify most of this


def autograd_backend():
    """Check if autograd is set as backend."""
    return gs.__name__.endswith("autograd")


def np_backend():
    """Check if numpy is set as backend."""
    return gs.__name__.endswith("numpy")


def pytorch_backend():
    """Check if pytorch is set as backend."""
    return gs.__name__.endswith("pytorch")


def autodiff_backend():
    return not np_backend()


if pytorch_backend():
    import torch


autograd_only = pytest.mark.skipif(
    not autograd_backend(), reason="Test for autograd backend only."
)
np_only = pytest.mark.skipif(not np_backend(), reason="Test for numpy backend only.")
torch_only = pytest.mark.skipif(
    not pytorch_backend(), reason="Test for pytorch backends only."
)


np_and_torch_only = pytest.mark.skipif(
    not (np_backend() or pytorch_backend()),
    reason="Test for numpy and pytorch backends only.",
)
np_and_autograd_only = pytest.mark.skipif(
    not (np_backend() or autograd_backend()),
    reason="Test for numpy and autograd backends only.",
)
autograd_and_torch_only = pytest.mark.skipif(
    not (autograd_backend() or pytorch_backend()),
    reason="Test for autograd and torch backends only.",
)


np_autograd_and_torch_only = pytest.mark.skipif(
    not (np_backend() or autograd_backend() or pytorch_backend()),
    reason="Test for numpy, autograd and pytorch backends only.",
)

autodiff_only = autograd_and_torch_only


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


def assert_allclose(a, b, rtol=gs.rtol, atol=gs.atol):
    if np_backend() or autograd_backend():
        return np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

    return assert_true(
        gs.allclose(a, b, rtol=rtol, atol=atol),
        msg=pytorch_error_msg(a, b, rtol, atol),
    )


def assert_true(condition, msg=None):
    assert condition, msg


class TestCase:
    """Class for Geomstats tests."""

    def _test_vectorization(self, vec_data, test_fnc_name=None):
        if test_fnc_name is None:
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            test_fnc_name = calframe[1][3][:-4]

        test_fnc = getattr(self, test_fnc_name)

        for datum in vec_data:
            test_fnc(**datum)

    def assertAllClose(self, a, b, rtol=gs.rtol, atol=gs.atol):
        if (gs.is_array(a) and gs.ndim(a) > 0) or (gs.is_array(b) and gs.ndim(b) > 0):
            self.assertShapeEqual(a, b)
        return assert_allclose(a, b, rtol=rtol, atol=atol)

    def assertAllEqual(self, a, b):
        if np_backend() or autograd_backend():
            np.testing.assert_array_equal(a, b)
        else:
            self.assertTrue(gs.all(gs.equal(a, b)))

    @staticmethod
    def assertTrue(condition, msg=None):
        return assert_true(condition, msg=msg)

    @staticmethod
    def assertFalse(condition, msg=None):
        assert not condition, msg

    @staticmethod
    def assertEqual(a, b, msg=None):
        assert a == b, msg

    @staticmethod
    def assertAllCloseToNp(a, np_a, rtol=gs.rtol, atol=gs.atol):
        are_same_shape = np.all(a.shape == np_a.shape)
        are_same = np.allclose(a, np_a, rtol=rtol, atol=atol)
        assert are_same and are_same_shape

    @staticmethod
    def assertShapeEqual(a, b):
        assert a.shape == b.shape
