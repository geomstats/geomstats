"""Regression tests for finite-difference arithmetic dtypes."""

import pytest

import geomstats.backend as gs
from geomstats.numerics.finite_differences import (
    centered_difference,
    forward_difference,
    second_centered_difference,
)
from geomstats.test.test_case import autodiff_only


@pytest.mark.parametrize(
    "function, values, dtype, expected, kwargs",
    [
        (forward_difference, [3, 2, 0], gs.uint8, [-1.0, -2.0], {}),
        (centered_difference, [3, 2, 0], gs.uint8, [-1.5], {}),
        (
            centered_difference,
            [3, 2, 0],
            gs.uint8,
            [-1.0, -1.5, -2.0],
            {"endpoints": True},
        ),
        (second_centered_difference, [3, 2, 0], gs.uint8, [-1.0], {}),
        (forward_difference, [-(2**30), 2**30], gs.int32, [float(2**31)], {}),
        (
            centered_difference,
            [-(2**30), 0, 2**30],
            gs.int32,
            [float(2**30)],
            {},
        ),
        (
            second_centered_difference,
            [0, 2**30 + 1, 0],
            gs.int32,
            [float(-(2**31) - 2)],
            {},
        ),
        (forward_difference, [True, True, False], bool, [0.0, -1.0], {}),
        (centered_difference, [True, True, False], bool, [-0.5], {}),
        (second_centered_difference, [True, True, False], bool, [-1.0], {}),
    ],
)
def test_integer_finite_differences_do_not_wrap(
    function, values, dtype, expected, kwargs
):
    """Cast before arithmetic can overflow or wrap unsigned differences."""
    array = gs.array(values, dtype=dtype)
    original = gs.copy(array)

    result = function(array, delta=1.0, **kwargs)

    assert gs.allclose(result, gs.array(expected))
    assert result.dtype == gs.get_default_dtype()
    assert array.dtype == original.dtype
    assert gs.all(array == original)


@pytest.mark.parametrize(
    "function, expected",
    [
        (forward_difference, [-2.0, -4.0]),
        (centered_difference, [-3.0]),
        (second_centered_difference, [-4.0]),
    ],
)
def test_integer_finite_differences_keep_batch_and_coordinate_axes(function, expected):
    """Differentiate quantized paths along the requested non-final axis."""
    values = gs.array([3, 2, 0], dtype=gs.uint8)
    array = gs.broadcast_to(gs.reshape(values, (1, 3, 1)), (2, 3, 4))

    result = function(array, axis=-2)
    expected = gs.broadcast_to(
        gs.reshape(gs.array(expected), (1, len(expected), 1)), (2, len(expected), 4)
    )

    assert result.shape == expected.shape
    assert gs.allclose(result, expected)


@pytest.mark.parametrize(
    "function, expected",
    [
        (forward_difference, [2.0, 3.0]),
        (centered_difference, [2.5]),
        (second_centered_difference, [1.0]),
    ],
)
@pytest.mark.parametrize("dtype", [gs.float32, gs.float64, gs.complex64, gs.complex128])
def test_finite_differences_preserve_inexact_input_dtypes(function, expected, dtype):
    """Preserve floating precision and the imaginary part of complex values."""
    array = gs.array([1.0, 3.0, 6.0], dtype=dtype)
    expected = gs.array(expected, dtype=dtype)
    if gs.is_complex(array):
        array = array * (1.0 + 2.0j)
        expected = expected * (1.0 + 2.0j)

    result = function(array, delta=1.0)

    assert result.dtype == array.dtype
    assert gs.allclose(result, expected)


@autodiff_only
@pytest.mark.parametrize(
    "function, expected",
    [
        (forward_difference, [[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]]),
        (centered_difference, [[-0.5, 0.0, 0.5]]),
        (second_centered_difference, [[1.0, -2.0, 1.0]]),
    ],
)
def test_finite_difference_dtype_handling_preserves_derivatives(function, expected):
    """Keep finite-difference path energies differentiable for floating data."""
    jacobian = gs.autodiff.jacobian(lambda array: function(array, delta=1.0))

    result = jacobian(gs.array([1.0, 3.0, 6.0]))

    assert gs.allclose(result, gs.array(expected))
