"""Regression tests for pointwise derivatives over several batch axes."""

import math

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.test.test_case import autodiff_only, autograd_only, pytorch_backend

pytestmark = autodiff_only


@pytest.mark.parametrize("point_shape", [(), (3,), (2, 2)])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize(
    "operator", ["jacobian_vec", "value_and_jacobian", "hessian_vec"]
)
def test_pointwise_derivative_preserves_batch_shape(point_shape, batch_shape, operator):
    """Compare each derivative with the analytic diagonal of a cubic."""
    size = math.prod(point_shape)
    points = (
        np.arange(math.prod(batch_shape) * size, dtype=float).reshape(
            batch_shape + point_shape
        )
        / 10.0
        + 0.5
    )
    flat_points = points.reshape((-1, size))
    if operator == "hessian_vec":
        function = lambda point: gs.sum(point**3)
        diagonals = 6.0 * flat_points
    else:
        function = lambda point: point**3
        diagonals = 3.0 * flat_points**2
    expected = np.stack([np.diag(diagonal) for diagonal in diagonals]).reshape(
        batch_shape + point_shape + point_shape
    )

    if operator == "value_and_jacobian" and pytorch_backend():
        if len(point_shape) != 1:
            pytest.skip("PyTorch value_and_jacobian only supports vector points.")
        derivative = gs.autodiff.value_and_jacobian(function)
    else:
        derivative = getattr(gs.autodiff, operator)(
            function, point_ndim=len(point_shape)
        )
    result = derivative(gs.array(points))
    if operator == "value_and_jacobian":
        value, result = result
        assert value.shape == points.shape
        assert gs.allclose(value, gs.array(points**3))

    assert result.shape == expected.shape
    assert gs.allclose(result, gs.array(expected))


@pytest.mark.parametrize("operator", ["jacobian_vec", "hessian_vec"])
@autograd_only
def test_autograd_batched_derivatives_remain_differentiable(operator):
    """Retain nested derivatives through the batch flattening and reshaping."""
    points = gs.reshape(gs.linspace(0.1, 0.6, 12), (2, 2, 3))
    if operator == "hessian_vec":
        function = lambda point: gs.sum(point**3)
        expected = 6.0 * gs.ones_like(points)
    else:
        function = lambda point: point**3
        expected = 6.0 * points
    derivative = getattr(gs.autodiff, operator)(function)

    _, gradient = gs.autodiff.value_and_grad(lambda point: gs.sum(derivative(point)))(
        points
    )

    assert gradient.shape == points.shape
    assert gs.allclose(gradient, expected)


def test_poincare_metric_derivatives_preserve_multiple_batch_axes():
    """Use the analytic conformal metric derivative as a geometry oracle."""
    points = gs.reshape(gs.linspace(0.01, 0.12, 12), (2, 3, 2))
    denominator = (1.0 - gs.sum(points**2, axis=-1)) ** 3
    expected = gs.einsum(
        "...k,ij->...ijk", 16.0 * points / denominator[..., None], gs.eye(2)
    )

    result = PoincareBall(2).metric.inner_product_derivative_matrix(points)

    assert result.shape == (2, 3, 2, 2, 2)
    assert gs.allclose(result, expected)
