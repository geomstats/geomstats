"""Regression tests for broadcasting endpoints in numerical geodesic solvers."""

import math

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.numerics.geodesic import (
    LogODESolver,
    LogShootingSolver,
    MultiresPathStraightening,
    PathStraightening,
)
from geomstats.numerics.optimization import ScipyMinimize

BATCH_SHAPES = [
    ((), ()),
    ((), (2,)),
    ((2,), ()),
    ((2,), (1,)),
    ((1,), (2,)),
    ((2, 1), (1, 3)),
    ((2, 1), (3,)),
]


def _endpoints(space, point_batch, base_batch):
    point_shape = point_batch + space.shape
    base_shape = base_batch + space.shape
    point = gs.reshape(gs.arange(math.prod(point_shape)), point_shape) / 20.0 + 1.0
    base_point = gs.reshape(gs.arange(math.prod(base_shape)), base_shape) / 10.0
    return point, base_point


def _solver(name, space):
    optimizer = ScipyMinimize()
    if name == "ode":
        return LogODESolver(space, n_nodes=5, use_jac=False)
    if name == "path":
        return PathStraightening(space, n_nodes=5, optimizer=optimizer)
    if name == "multires":
        return MultiresPathStraightening(space, n_nodes=[3, 5], optimizer=optimizer)
    return LogShootingSolver(
        space, flatten=name == "shooting_flat", optimizer=optimizer
    )


@pytest.mark.parametrize("point_batch,base_batch", BATCH_SHAPES)
@pytest.mark.parametrize(
    "solver_name,space",
    [
        (name, space)
        for name in ["ode", "shooting", "shooting_flat", "path"]
        for space in [Euclidean(2), Matrices(2, 2)]
        if name != "ode" or space.point_ndim == 1
    ],
)
def test_log_solver_broadcasts_all_endpoint_batch_axes(
    space, point_batch, base_batch, solver_name
):
    """Compare every broadcasted tangent vector with the closed-form log."""
    point, base_point = _endpoints(space, point_batch, base_batch)
    expected = point - base_point

    result = _solver(solver_name, space).log(point, base_point)

    assert result.shape == expected.shape
    assert gs.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("point_batch,base_batch", BATCH_SHAPES)
@pytest.mark.parametrize(
    "solver_name,space",
    [("ode", Euclidean(2)), ("path", Euclidean(2)), ("path", Matrices(2, 2))],
)
def test_bvp_geodesic_preserves_broadcasted_batch_and_time_axes(
    space, point_batch, base_batch, solver_name
):
    """Return all endpoint pairs, keeping the time axis after every batch axis."""
    point, base_point = _endpoints(space, point_batch, base_batch)
    expected_point, expected_base = gs.broadcast_arrays(point, base_point)
    batch_shape = expected_point.shape[: -space.point_ndim]
    path = _solver(solver_name, space).geodesic_bvp(point, base_point)

    for times in (gs.array([0.0, 0.25, 1.0]), gs.array([0.5])):
        result = path(times)
        time_shape = (1,) * len(batch_shape) + (-1,) + (1,) * space.point_ndim
        expected = gs.expand_dims(expected_base, axis=-space.point_ndim - 1) + (
            gs.reshape(times, time_shape)
            * gs.expand_dims(expected_point - expected_base, axis=-space.point_ndim - 1)
        )

        assert result.shape == expected.shape
        assert gs.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("space", [Euclidean(2), Matrices(2, 2)])
def test_multires_path_straightening_preserves_broadcasted_endpoint_pairs(space):
    """Exercise the shared batching mixin with a multiresolution solver."""
    point, base_point = _endpoints(space, (2, 1), (1, 3))
    point_broadcast, base_broadcast = gs.broadcast_arrays(point, base_point)

    path = _solver("multires", space).discrete_geodesic_bvp(point, base_point)

    assert path.shape == (2, 3, 5) + space.shape
    point_axes = (slice(None),) * space.point_ndim
    assert gs.allclose(path[(..., 0) + point_axes], base_broadcast)
    assert gs.allclose(path[(..., -1) + point_axes], point_broadcast)
    assert gs.allclose(
        path[(..., 2) + point_axes], (point_broadcast + base_broadcast) / 2.0
    )
