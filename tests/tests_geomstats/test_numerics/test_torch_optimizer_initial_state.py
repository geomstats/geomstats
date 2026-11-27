"""Regression tests for independent initial states in torch optimizers."""

import pytest

from geomstats.test.test_case import torch_only

torch = pytest.importorskip("torch")

from geomstats.numerics import optimization  # noqa: E402

pytestmark = torch_only


def make_optimizer(name):
    """Use short, deterministic optimization runs for each wrapper family."""
    if name == "TorchLBFGS":
        return optimization.TorchLBFGS(max_iter=5, save_result=True)
    return getattr(optimization, name)(max_iter=5, lr=0.1, tol=0.0, save_result=True)


@pytest.mark.parametrize(
    "name", ["TorchLBFGS", "TorchSGD", "TorchAdam", "TorchRMSprop"]
)
@pytest.mark.parametrize("initial_kind", ["plain", "leaf", "nonleaf"])
def test_torch_optimizer_copies_differentiable_initial_points(name, initial_kind):
    """Optimize values without modifying the caller's tensor or its graph."""
    parent = torch.tensor(
        [3.0, -2.0], dtype=torch.float64, requires_grad=initial_kind != "plain"
    )
    initial = parent * 0.5 if initial_kind == "nonleaf" else parent
    original = initial.detach().clone()
    parent_original = parent.detach().clone()
    if parent.requires_grad:
        parent.grad = torch.tensor([7.0, -3.0], dtype=parent.dtype)
        original_gradient = parent.grad.clone()

    target = torch.tensor([1.0, 0.5], dtype=initial.dtype)

    def objective(point):
        return ((point - target) ** 2).sum()

    optimizer = make_optimizer(name)
    result = optimizer.minimize(objective, initial)

    assert objective(result.x) < objective(original)
    assert result.x.is_leaf
    assert result.x.requires_grad
    assert result.x.data_ptr() != initial.data_ptr()
    assert torch.equal(initial.detach(), original)
    assert torch.equal(parent.detach(), parent_original)
    if parent.requires_grad:
        assert torch.equal(parent.grad, original_gradient)
    assert optimizer.result_ is result


@pytest.mark.parametrize(
    "name", ["TorchLBFGS", "TorchSGD", "TorchAdam", "TorchRMSprop"]
)
def test_torch_optimizer_can_restart_from_its_previous_result(name):
    """Reuse a returned differentiable point as the next initial guess."""
    optimizer = make_optimizer(name)
    first = optimizer.minimize(
        lambda point: ((point - 1.0) ** 2).sum(),
        torch.tensor([3.0, -2.0], dtype=torch.float64),
    )
    previous = first.x.detach().clone()

    def next_objective(point):
        return ((point + 2.0) ** 2).sum()

    second = optimizer.minimize(next_objective, first.x)

    assert next_objective(second.x) < next_objective(previous)
    assert second.x.is_leaf
    assert second.x.data_ptr() != first.x.data_ptr()
    assert torch.equal(first.x.detach(), previous)
    assert optimizer.result_ is second
