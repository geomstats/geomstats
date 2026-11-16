"""Regression tests for per-point gradients of broadcasted input batches."""

import math

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.test.test_case import autodiff_only, pytorch_backend

pytestmark = autodiff_only

BATCH_SHAPES = [
    ((), ()),
    ((1,), (3,)),
    ((3,), (1,)),
    ((), (3,)),
    ((3,), ()),
    ((2, 1), (1, 3)),
    ((1, 3), (2, 1)),
    ((2, 1, 1), (1, 3)),
]


def _values(shape, offset):
    return offset + gs.reshape(gs.arange(math.prod(shape)), shape) / 100.0


def _skip_existing_torch_output_limitation(first_batch, second_batch):
    if pytorch_backend() and max(len(first_batch), len(second_batch)) > 1:
        pytest.skip("PyTorch multi-axis output differentiation is addressed in #2239.")


@pytest.mark.parametrize("first_batch,second_batch", BATCH_SHAPES)
@pytest.mark.parametrize("point_shape", [(), (2,), (2, 2)])
def test_value_and_grad_broadcasts_all_point_pairs(
    point_shape, first_batch, second_batch
):
    """Return one derivative per broadcasted pair without summing over shared inputs."""
    _skip_existing_torch_output_limitation(first_batch, second_batch)
    first = _values(first_batch + point_shape, 0.2)
    second = _values(second_batch + point_shape, 0.1)
    point_ndim = len(point_shape)
    axes = tuple(range(-point_ndim, 0))

    def objective(x, y):
        squared = (x - y) ** 2
        return gs.sum(squared, axis=axes) if point_ndim else squared

    value, gradients = gs.autodiff.value_and_grad(
        objective, argnums=(0, 1), point_ndims=point_ndim
    )(first, second)

    first_broadcast, second_broadcast = gs.broadcast_arrays(first, second)
    difference = first_broadcast - second_broadcast
    expected_value = objective(first_broadcast, second_broadcast)
    assert value.shape == expected_value.shape
    assert gs.allclose(value, expected_value)
    assert gradients[0].shape == gradients[1].shape == difference.shape
    assert gs.allclose(gradients[0], 2.0 * difference)
    assert gs.allclose(gradients[1], -2.0 * difference)


@pytest.mark.parametrize("first_batch,second_batch", BATCH_SHAPES)
@pytest.mark.parametrize("point_shape", [(2,), (2, 2)])
def test_value_and_grad_broadcasts_mixed_point_ranks(
    point_shape, first_batch, second_batch
):
    """Keep a scalar point's batch axes separate from vector or matrix coordinates."""
    _skip_existing_torch_output_limitation(first_batch, second_batch)
    first = _values(first_batch + point_shape, 0.2)
    second = _values(second_batch, 0.1)
    point_ndim = len(point_shape)
    axes = tuple(range(-point_ndim, 0))

    def objective(x, y):
        y = gs.reshape(y, y.shape + (1,) * point_ndim)
        return gs.sum((x - y) ** 2, axis=axes)

    value, gradients = gs.autodiff.value_and_grad(
        objective, argnums=(0, 1), point_ndims=(point_ndim, 0)
    )(first, second)

    expanded_second = gs.reshape(second, second.shape + (1,) * point_ndim)
    difference = first - expanded_second
    expected_value = gs.sum(difference**2, axis=axes)
    assert value.shape == expected_value.shape
    assert gs.allclose(value, expected_value)
    assert gradients[0].shape == difference.shape
    assert gradients[1].shape == expected_value.shape
    assert gs.allclose(gradients[0], 2.0 * difference)
    assert gs.allclose(gradients[1], -2.0 * gs.sum(difference, axis=axes))


def test_value_and_grad_pairwise_euclidean_squared_distances():
    """A singleton input represents a separate point in every distance pair."""
    space = Euclidean(2)
    first = gs.array([[0.1, 0.2]])
    second = gs.array([[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])

    value, gradients = gs.autodiff.value_and_grad(
        space.metric.squared_dist, argnums=(0, 1)
    )(first, second)

    difference = first - second
    assert value.shape == (3,)
    assert gs.allclose(value, gs.sum(difference**2, axis=-1))
    assert gradients[0].shape == gradients[1].shape == (3, 2)
    assert gs.allclose(gradients[0], 2.0 * difference)
    assert gs.allclose(gradients[1], -2.0 * difference)
