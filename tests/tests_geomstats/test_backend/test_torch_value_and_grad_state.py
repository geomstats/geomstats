"""Regression tests for PyTorch gradient evaluation without caller mutation."""

import math

import pytest

import geomstats.backend as gs
from geomstats.test.test_case import torch_only

torch = pytest.importorskip("torch")
pytestmark = torch_only


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 2), (2, 1, 3)])
def test_value_and_grad_ignores_and_preserves_accumulated_gradients(batch_shape):
    """Repeated evaluations compute fresh gradients for every batch point."""
    points = torch.linspace(0.1, 0.9, math.prod(batch_shape) * 3, dtype=torch.float64)
    points = points.reshape(batch_shape + (3,)).requires_grad_(True)
    points.grad = torch.full_like(points, 0.7)
    previous_grad = points.grad.clone()
    previous_value = points.detach().clone()

    for scale in [1.0, 0.5]:
        value, gradient = gs.autodiff.value_and_grad(
            lambda point: scale * (point**2).sum(dim=-1)
        )(points)

        assert torch.allclose(value, scale * (previous_value**2).sum(dim=-1))
        assert gradient.shape == points.shape
        assert torch.allclose(gradient, 2.0 * scale * previous_value)
        assert torch.equal(points.grad, previous_grad)
        assert torch.equal(points.detach(), previous_value)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_value_and_grad_accepts_nonleaf_inputs(batch_shape):
    """Differentiate with respect to the supplied point without touching ancestors."""
    parent = torch.linspace(0.1, 0.9, math.prod(batch_shape) * 3, dtype=torch.float64)
    parent = parent.reshape(batch_shape + (3,)).requires_grad_(True)
    parent.grad = torch.full_like(parent, 2.0)
    points = parent * 2.0

    value, gradient = gs.autodiff.value_and_grad(lambda point: (point**3).sum(dim=-1))(
        points
    )

    assert torch.allclose(value, (points.detach() ** 3).sum(dim=-1))
    assert torch.allclose(gradient, 3.0 * points.detach() ** 2)
    assert torch.equal(parent.grad, torch.full_like(parent, 2.0))


@pytest.mark.parametrize("argnums", [0, 1, -1, (0,), (1,), (1, 0), (0, 1)])
def test_value_and_grad_preserves_requested_argument_order(argnums):
    """Return only the requested derivatives, in the requested order."""
    first = torch.tensor([1.0, 2.0], requires_grad=True)
    second = torch.tensor([3.0, 4.0], requires_grad=True)
    first.grad = torch.full_like(first, 5.0)
    second.grad = torch.full_like(second, 7.0)

    value, gradient = gs.autodiff.value_and_grad(
        lambda x, y: (x**2 + 3.0 * y**2).sum(), argnums=argnums
    )(first, second)

    expected = (2.0 * first.detach(), 6.0 * second.detach())
    if isinstance(argnums, int):
        assert torch.allclose(gradient, expected[argnums])
    else:
        assert len(gradient) == len(argnums)
        for actual, index in zip(gradient, argnums):
            assert torch.allclose(actual, expected[index])
    assert torch.allclose(
        value, (first.detach() ** 2 + 3.0 * second.detach() ** 2).sum()
    )
    assert torch.equal(first.grad, torch.full_like(first, 5.0))
    assert torch.equal(second.grad, torch.full_like(second, 7.0))


def test_value_and_grad_returns_zero_for_an_unused_argument():
    """The derivative with respect to an independent argument is zero."""
    first = torch.tensor([1.0, 2.0])
    second = torch.tensor([3.0, 4.0])

    value, gradients = gs.autodiff.value_and_grad(
        lambda x, y: (x**2).sum(), argnums=(0, 1)
    )(first, second)

    assert torch.allclose(value, (first**2).sum())
    assert torch.allclose(gradients[0], 2.0 * first)
    assert torch.equal(gradients[1], torch.zeros_like(second))
    assert first.grad is None
    assert second.grad is None


def test_value_and_grad_without_selected_arguments():
    """An empty selection returns the value without accumulating gradients."""
    point = torch.tensor([1.0, 2.0], requires_grad=True)

    value, gradients = gs.autodiff.value_and_grad(lambda x: (x**2).sum(), argnums=())(
        point
    )

    assert torch.allclose(value, torch.tensor(5.0))
    assert gradients == ()
    assert point.grad is None
