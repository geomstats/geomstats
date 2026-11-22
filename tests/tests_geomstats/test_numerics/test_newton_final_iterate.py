"""Regression tests for Newton convergence and evaluation counts at the limit."""

import pytest

import geomstats.backend as gs
from geomstats.numerics.optimization import NewtonMethod


@pytest.mark.parametrize("max_iter", [0, 1, 3])
def test_newton_initial_root_needs_no_jacobian(max_iter):
    """Recognize an initial root, including when no updates are permitted."""
    calls = []

    def fun(point):
        calls.append(gs.copy(point))
        return point - gs.array([1.0, 2.0])

    def jacobian(point):
        pytest.fail("An initial root does not require a Jacobian evaluation.")

    result = NewtonMethod(max_iter=max_iter).root(fun, gs.array([1.0, 2.0]), jacobian)

    assert result.success
    assert result.status == 1
    assert result.nfev == len(calls) == 1
    assert result.njac == 0
    assert gs.allclose(result.x, gs.array([1.0, 2.0]))


def test_newton_reports_convergence_on_last_permitted_update():
    """One Newton update solves a nonsingular linear system exactly."""
    matrix = gs.array([[3.0, 1.0], [1.0, 2.0]])
    target = gs.array([1.0, -2.0])
    function_calls = []
    jacobian_calls = []

    def fun(point):
        function_calls.append(gs.copy(point))
        return gs.matvec(matrix, point) - target

    def jacobian(point):
        jacobian_calls.append(gs.copy(point))
        return matrix

    result = NewtonMethod(max_iter=1).root(fun, gs.zeros(2), jacobian)

    assert result.success
    assert result.status == 1
    assert result.nfev == len(function_calls) == 2
    assert result.njac == len(jacobian_calls) == 1
    assert gs.allclose(result.x, gs.array([0.8, -1.4]))
    assert gs.allclose(function_calls[-1], result.x)


@pytest.mark.parametrize("max_iter,success", [(0, False), (1, False), (2, True)])
def test_newton_checks_final_nonlinear_residual(max_iter, success):
    """Use the residual at the returned point to report convergence."""
    function_calls = []
    jacobian_calls = []

    def fun(point):
        function_calls.append(gs.copy(point))
        return point**2 - 2.0

    def jacobian(point):
        jacobian_calls.append(gs.copy(point))
        return gs.reshape(2.0 * point, (1, 1))

    result = NewtonMethod(atol=0.01, max_iter=max_iter).root(
        fun, gs.array([1.0]), jacobian
    )

    assert result.success == success
    assert result.status == int(success)
    assert result.nfev == len(function_calls) == max_iter + 1
    assert result.njac == len(jacobian_calls) == max_iter
    assert gs.allclose(function_calls[-1], result.x)
    assert bool(gs.linalg.vector_norm(result.x**2 - 2.0) <= 0.01) == success


def test_damped_newton_checks_convergence_after_its_last_update():
    """Apply the same final-point convergence check to damped Newton updates."""
    result = NewtonMethod(atol=0.6, max_iter=1, damped=True).root(
        lambda point: point - 1.0,
        gs.array([0.0]),
        lambda point: gs.eye(1),
    )

    assert result.success
    assert gs.allclose(result.x, gs.array([0.5]))
    assert result.nfev == 2
    assert result.njac == 1
