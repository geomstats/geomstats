"""Regression tests for connection-barycenter backtracking."""

import math

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.learning.exponential_barycenter import (
    ConnectionGradientDescent,
    ExponentialBarycenter,
)


@pytest.mark.parametrize("step_size", [1.0, 2.0, 3.0, 4.0])
def test_connection_barycenter_backtracks_after_overshooting(step_size):
    """Recover the arithmetic mean even when the initial step overshoots it."""
    space = Euclidean(1)
    points = gs.array([[0.0], [2.0]])
    optimizer = ConnectionGradientDescent(
        init_step_size=step_size, max_iter=100, epsilon=1e-10
    )

    result = optimizer.minimize(space, points)

    assert gs.allclose(result, gs.array([1.0]), atol=1e-8)


@pytest.mark.parametrize("weight_scale", [1.0, 10.0])
def test_connection_barycenter_backtracking_preserves_weighted_mean(weight_scale):
    """Retain weight normalization and a user-supplied starting point."""
    space = Euclidean(2)
    points = gs.array([[0.0, 0.0], [2.0, 4.0], [4.0, 2.0]])
    weights = weight_scale * gs.array([1.0, 2.0, 1.0])
    optimizer = ConnectionGradientDescent(
        init_point=gs.array([10.0, -4.0]),
        init_step_size=3.0,
        max_iter=100,
        epsilon=1e-10,
    )

    result = optimizer.minimize(space, points, weights=weights)

    assert gs.allclose(result, gs.array([2.0, 2.5]), atol=1e-8)
    assert gs.all(weights == weight_scale * gs.array([1.0, 2.0, 1.0]))


def test_connection_barycenter_backtracking_accepts_matrix_points():
    """Keep all matrix coordinates when computing the mean logarithm."""
    space = Matrices(2, 2)
    points = gs.array([[[0.0, 2.0], [1.0, 3.0]], [[4.0, 0.0], [7.0, 1.0]]])
    optimizer = ConnectionGradientDescent(init_step_size=4.0)

    result = optimizer.minimize(space, points)

    assert result.shape == space.shape
    assert gs.allclose(result, gs.array([[2.0, 1.0], [4.0, 2.0]]))


def test_connection_barycenter_keeps_last_accepted_point_on_iteration_limit():
    """A rejected trial must not replace the current estimate."""
    space = Euclidean(1)
    initial_point = gs.array([0.0])
    optimizer = ConnectionGradientDescent(
        init_point=initial_point, init_step_size=3.0, max_iter=1
    )

    result = optimizer.minimize(space, gs.array([[0.0], [2.0]]))

    assert gs.allclose(result, initial_point)


def test_exponential_barycenter_recovers_spherical_geodesic_midpoint():
    """Exercise the public estimator on a curved space with a known midpoint."""
    space = Hypersphere(2)
    angle = 0.4
    points = gs.array([[1.0, 0.0, 0.0], [math.cos(angle), math.sin(angle), 0.0]])
    estimator = ExponentialBarycenter(
        space,
        optimizer_kwargs={"init_step_size": 3.0, "max_iter": 100, "epsilon": 1e-10},
    )

    result = estimator.fit(points).estimate_
    expected = gs.array([math.cos(angle / 2), math.sin(angle / 2), 0.0])

    assert gs.allclose(result, expected, atol=1e-8)
    assert space.belongs(result)
