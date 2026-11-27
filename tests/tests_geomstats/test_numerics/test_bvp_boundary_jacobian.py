"""Regression tests for backend-aware boundary Jacobian callbacks."""

import pytest

import geomstats.backend as gs
from geomstats.numerics.bvp import ScipySolveBVP


@pytest.mark.parametrize("use_fun_jac", [False, True])
def test_bvp_boundary_jacobian_receives_backend_states(use_fun_jac):
    """Solve a linear boundary problem with analytic Jacobians in the backend."""
    boundary_calls = []

    def fun(time, state):
        return gs.stack([state[1], gs.zeros_like(state[0])])

    def boundary(state_0, state_1):
        return gs.stack([state_0[0], state_1[0] - 1.0])

    def boundary_jacobian(state_0, state_1):
        boundary_calls.append((state_0, state_1))
        zeros_0 = gs.zeros_like(state_0)
        zeros_1 = gs.zeros_like(state_1)
        first_0 = zeros_0 + gs.array([1.0, 0.0])
        first_1 = zeros_1 + gs.array([1.0, 0.0])
        return gs.stack([first_0, zeros_0]), gs.stack([zeros_1, first_1])

    def fun_jacobian(time, state):
        zeros = gs.zeros_like(state[0])
        return gs.stack([gs.stack([zeros, zeros + 1.0]), gs.stack([zeros, zeros])])

    mesh = gs.linspace(0.0, 1.0, 5)
    solver = ScipySolveBVP(save_result=True)
    result = solver.integrate(
        fun,
        boundary,
        mesh,
        gs.zeros((2, 5)),
        fun_jac=fun_jacobian if use_fun_jac else None,
        bc_jac=boundary_jacobian,
    )

    assert boundary_calls
    assert all(gs.is_array(state) for pair in boundary_calls for state in pair)
    assert result.success
    assert gs.allclose(result.y[0], result.x)
    assert gs.allclose(result.y[1], gs.ones_like(result.x))
    assert solver.result_ is result


def test_bvp_without_boundary_jacobian_still_uses_numerical_derivatives():
    """Retain SciPy's numerical-Jacobian path when no callback is supplied."""

    def fun(time, state):
        return gs.stack([state[1], gs.zeros_like(state[0])])

    def boundary(state_0, state_1):
        return gs.stack([state_0[0], state_1[0] - 1.0])

    result = ScipySolveBVP().integrate(
        fun, boundary, gs.linspace(0.0, 1.0, 5), gs.zeros((2, 5))
    )

    assert result.success
    assert gs.allclose(result.y[0], result.x)
    assert gs.allclose(result.y[1], gs.ones_like(result.x))
