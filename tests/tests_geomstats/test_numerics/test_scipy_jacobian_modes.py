"""Regression tests for SciPy numerical and combined Jacobian modes."""

import pytest

import geomstats.backend as gs
from geomstats.numerics.optimization import ScipyMinimize
from geomstats.test.test_case import autodiff_only


def _quadratic(point):
    delta = point - gs.array([1.25, -0.75])
    return gs.sum(gs.array([1.0, 3.0]) * delta**2)


def _gradient(point):
    return 2.0 * gs.array([1.0, 3.0]) * (point - gs.array([1.25, -0.75]))


@pytest.mark.parametrize("jacobian", [None, False, "2-point", "3-point", "cs"])
def test_minimize_scalar_objective_with_numerical_jacobian(jacobian):
    """Allow each SciPy numerical-gradient mode to optimize a scalar function."""
    optimizer = ScipyMinimize(method="BFGS", tol=1e-6, save_result=True)

    result = optimizer.minimize(_quadratic, gs.array([-2.0, 2.0]), fun_jac=jacobian)

    assert result.success
    assert gs.allclose(result.x, gs.array([1.25, -0.75]), atol=1e-5)
    assert result.fun < 1e-10
    assert optimizer.result_ is result


@pytest.mark.parametrize("combined", [False, True])
def test_minimize_preserves_callable_and_combined_jacobians(combined):
    """Keep both ways of supplying analytic gradients."""
    optimizer = ScipyMinimize(method="BFGS", tol=1e-6)
    objective = (
        (lambda point: (_quadratic(point), _gradient(point)))
        if combined
        else _quadratic
    )
    jacobian = True if combined else _gradient

    result = optimizer.minimize(objective, gs.array([-2.0, 2.0]), fun_jac=jacobian)

    assert result.success
    assert gs.allclose(result.x, gs.array([1.25, -0.75]), atol=1e-5)
    assert result.fun < 1e-10


@autodiff_only
@pytest.mark.parametrize("jacobian", [False, "2-point", "3-point", "cs"])
def test_numerical_jacobian_can_be_combined_with_automatic_hessian(jacobian):
    """Differentiate the scalar objective, not a nonexistent tuple element."""
    optimizer = ScipyMinimize(
        method="trust-constr", autodiff_jac=True, autodiff_hess=True, tol=1e-6
    )

    result = optimizer.minimize(_quadratic, gs.array([-2.0, 2.0]), fun_jac=jacobian)

    assert result.success
    assert gs.allclose(result.x, gs.array([1.25, -0.75]), atol=1e-5)
    assert result.fun < 1e-10


@autodiff_only
def test_combined_objective_gradient_still_supports_automatic_hessian():
    """Extract only the value when the objective explicitly returns a tuple."""
    optimizer = ScipyMinimize(method="trust-exact", autodiff_hess=True, tol=1e-6)

    result = optimizer.minimize(
        lambda point: (_quadratic(point), _gradient(point)),
        gs.array([-2.0, 2.0]),
        fun_jac=True,
    )

    assert result.success
    assert gs.allclose(result.x, gs.array([1.25, -0.75]), atol=1e-5)
    assert result.fun < 1e-10
