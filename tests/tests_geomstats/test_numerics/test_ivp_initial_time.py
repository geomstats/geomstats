"""Regression tests for evaluating an IVP only at its initial instant."""

import math

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.numerics.geodesic import ExpODESolver
from geomstats.numerics.ivp import ScipySolveIVP


@pytest.mark.parametrize("method", ["RK45", "Radau", "BDF"])
@pytest.mark.parametrize("point_shape", [(3,), (2, 2)])
@pytest.mark.parametrize("batch_shape", [(), (1,), (3,)])
@pytest.mark.parametrize("time_kind", ["list", "tuple", "array"])
def test_ivp_initial_time_preserves_state_shape(
    method, point_shape, batch_shape, time_kind
):
    """Return one requested state for vector, matrix, and batched IVPs."""
    shape = batch_shape + (2,) + point_shape
    initial_state = 0.1 + gs.reshape(gs.arange(math.prod(shape)), shape) / 10.0
    original = gs.copy(initial_state)
    times = {"list": [0.0], "tuple": (0.0,), "array": gs.array([0.0])}[time_kind]
    solver = ScipySolveIVP(method=method, point_ndim=len(point_shape), save_result=True)

    result = solver.integrate_t(lambda state, time: -state, initial_state, times)

    expected = gs.expand_dims(initial_state, axis=len(batch_shape))
    assert result.t.shape == (1,)
    assert gs.allclose(result.t, gs.array([0.0]))
    assert result.y.shape == expected.shape
    assert gs.allclose(result.y, expected)
    assert gs.all(initial_state == original)
    assert solver.result_ is result
    if batch_shape:
        assert all(result.success)
        assert result.status == [0] * batch_shape[0]
    else:
        assert result.success
        assert result.status == 0


@pytest.mark.parametrize("batch_shape", [(), (2,)])
@pytest.mark.parametrize("time_kind", ["scalar", "zero_dim", "array"])
def test_numerical_geodesic_at_zero_is_its_base_point(batch_shape, time_kind):
    """A geodesic evaluated only at time zero contains the base point once."""
    space = Euclidean(3)
    shape = batch_shape + space.shape
    base_point = gs.reshape(gs.arange(math.prod(shape)), shape) / 10.0
    tangent_vec = gs.ones_like(base_point)
    solver = ExpODESolver(space, integrator=ScipySolveIVP())
    time = {"scalar": 0.0, "zero_dim": gs.array(0.0), "array": gs.array([0.0])}[
        time_kind
    ]

    result = solver.geodesic_ivp(tangent_vec, base_point)(time)

    expected = gs.expand_dims(base_point, axis=-2)
    assert result.shape == expected.shape
    assert gs.allclose(result, expected)


def test_ivp_initial_time_keeps_complex_state_and_dense_output():
    """Retain native solver dtype conversion and dense-output metadata."""
    initial_state = gs.array([[1.0 + 2.0j, 3.0 - 1.0j]])
    solver = ScipySolveIVP(dense_output=True)

    result = solver.integrate_t(
        lambda state, time: 1j * state, initial_state, gs.array([0.0])
    )

    assert result.y.shape == (1,) + initial_state.shape
    assert gs.is_complex(result.y)
    assert gs.allclose(result.y[0], initial_state)
    assert gs.allclose(gs.from_numpy(result.sol(0.0)), gs.reshape(initial_state, (-1,)))
    assert result.success


@pytest.mark.parametrize("method", ["RK45", "Radau", "BDF"])
def test_ivp_positive_interval_keeps_all_requested_times(method):
    """Leave ordinary integrations and their initial sample unchanged."""
    initial_state = gs.array([[1.0, 2.0], [3.0, 4.0]])
    solver = ScipySolveIVP(method=method)

    result = solver.integrate_t(
        lambda state, time: gs.ones_like(state), initial_state, gs.array([0.0, 0.25])
    )

    expected = gs.stack([initial_state, initial_state + 0.25])
    assert result.y.shape == expected.shape
    assert gs.allclose(result.y, expected)
    assert gs.allclose(result.t, gs.array([0.0, 0.25]))
    assert result.success


def test_ivp_zero_interval_retains_native_time_validation():
    """Do not bypass SciPy's rejection of evaluation times outside the interval."""
    solver = ScipySolveIVP()

    with pytest.raises(ValueError, match="t_eval"):
        solver._integrate_single(
            lambda state, time: state,
            gs.array([[1.0, 2.0]]),
            end_time=0.0,
            t_eval=gs.array([1.0]),
        )
