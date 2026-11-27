"""Regression tests for callable IVP steps and their solver results."""

import pytest

import geomstats.backend as gs
from geomstats.integrator import euler_step, rk2_step, rk4_step
from geomstats.numerics.ivp import GSIVPIntegrator
from geomstats.test.test_case import autodiff_only


@pytest.mark.parametrize(
    "step, n_evaluations", [(euler_step, 1), (rk2_step, 2), (rk4_step, 4)]
)
@pytest.mark.parametrize("n_steps", [1, 4])
def test_callable_ivp_step_returns_correct_trajectory_and_evaluation_count(
    step, n_evaluations, n_steps
):
    """Integrate a time-dependent force with a callable Runge-Kutta step."""
    initial_state = gs.zeros((2, 1, 3))
    solver = GSIVPIntegrator(n_steps=n_steps, step_type=step, save_result=True)

    result = solver.integrate(
        lambda state, time: (2.0 * time + 1.0) * gs.ones_like(state),
        initial_state,
        end_time=2.0,
    )
    expected = 6.0 - 4.0 / n_steps if step is euler_step else 6.0

    assert result.success
    assert result.nfev == n_evaluations * n_steps
    assert result.y.shape == (n_steps + 1,) + initial_state.shape
    assert gs.allclose(result.t, gs.linspace(0.0, 2.0, n_steps + 1))
    assert gs.allclose(result.get_last_y(), expected * gs.ones_like(initial_state))
    assert solver.result_ is result


def test_custom_ivp_step_counts_a_variable_number_of_force_evaluations():
    """Count actual calls instead of assuming a fixed number per step."""
    evaluations = []

    def force(state, time):
        evaluations.append(time)
        return gs.ones_like(state)

    def step(force, state, time, dt):
        derivative = force(state, time)
        if time > 0.0:
            force(state=state, time=time + dt)
        return state + dt * derivative

    solver = GSIVPIntegrator(n_steps=3, step_type=step)
    result = solver.integrate(force, gs.array([0.0]))

    assert result.nfev == len(evaluations) == 5
    assert gs.allclose(result.get_last_y(), gs.array([1.0]))
    assert solver.result_ is None


def test_custom_ivp_step_without_force_evaluations():
    """Allow exact custom updates that do not evaluate the supplied force."""

    def step(force, state, time, dt):
        return state + dt

    def force(state, time):
        pytest.fail("The custom step should not call the force.")

    solver = GSIVPIntegrator(n_steps=4, step_type=step)
    result = solver.integrate(force, gs.array([0.0]))

    assert result.nfev == 0
    assert result.success
    assert gs.allclose(result.get_last_y(), gs.array([1.0]))


def test_ivp_evaluation_count_resets_and_step_type_can_change():
    """Keep evaluation counts local to each integration and each step type."""
    solver = GSIVPIntegrator(n_steps=3, step_type=rk2_step)

    def force(state, time):
        return gs.ones_like(state)

    first = solver.integrate(force, gs.array([0.0]))
    second = solver.integrate(force, gs.array([0.0]), end_time=2.0)
    solver.step_type = "euler"
    third = solver.integrate(force, gs.array([0.0]))

    assert first.nfev == second.nfev == 6
    assert third.nfev == 3
    assert gs.allclose(first.get_last_y(), gs.array([1.0]))
    assert gs.allclose(second.get_last_y(), gs.array([2.0]))
    assert third.success


@pytest.mark.parametrize(
    "step_type, n_evaluations", [("euler", 1), ("rk2", 2), ("rk4", 4)]
)
def test_named_ivp_steps_expose_standard_success_metadata(step_type, n_evaluations):
    """Keep named-step evaluation counts and expose OdeResult.success."""
    solver = GSIVPIntegrator(n_steps=4, step_type=step_type)
    result = solver.integrate(lambda state, time: gs.ones_like(state), gs.array([0.0]))

    assert result.success
    assert result.nfev == 4 * n_evaluations
    assert gs.allclose(result.get_last_y(), gs.array([1.0]))


@autodiff_only
def test_callable_ivp_step_preserves_differentiation():
    """Differentiate a complete custom-step flow through its initial state."""
    n_steps = 4
    solver = GSIVPIntegrator(n_steps=n_steps, step_type=rk2_step)

    def endpoint(initial_state):
        return solver.integrate(lambda state, time: state, initial_state).get_last_y()

    jacobian = gs.autodiff.jacobian(endpoint)(gs.array([0.5, -0.1]))
    dt = 1.0 / n_steps
    expected = (1.0 + dt + dt**2 / 2.0) ** n_steps * gs.eye(2)

    assert gs.allclose(jacobian, expected)
