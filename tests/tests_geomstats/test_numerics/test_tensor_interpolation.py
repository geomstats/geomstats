"""Regression tests for interpolation of tensor-valued paths."""

import math

import pytest

import geomstats.backend as gs
from geomstats.numerics.interpolation import (
    LinearInterpolator1D,
    UniformUnitIntervalLinearInterpolator,
)
from geomstats.numerics.path import UniformlySampledDiscretePath


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize(
    "point_shape",
    [(), (2,), (2, 3), (2, 3, 2), (2, 3, 2, 2), (2, 1, 2, 3, 2)],
)
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 1)])
@pytest.mark.parametrize("query_times", [(0.2, 0.8), (0.125, 0.375, 0.875)])
def test_linear_interpolation_keeps_time_separate_from_point_axes(
    uniform, point_shape, batch_shape, query_times
):
    """Recover affine paths independently of their point and batch ranks."""
    shape = batch_shape + point_shape
    start = gs.reshape(gs.arange(math.prod(shape), dtype=gs.float64), shape)
    velocity = 0.5 + gs.abs(start)
    sample_times = gs.array([0.0, 0.5 if uniform else 0.25, 1.0])
    data = gs.stack(
        [start + time * velocity for time in sample_times], axis=len(batch_shape)
    )

    if uniform:
        interpolator = UniformUnitIntervalLinearInterpolator(
            data, point_ndim=len(point_shape)
        )
    else:
        interpolator = LinearInterpolator1D(
            sample_times, data, point_ndim=len(point_shape)
        )

    result = interpolator(gs.array(query_times))
    expected = gs.stack(
        [start + time * velocity for time in query_times], axis=len(batch_shape)
    )

    assert result.shape == batch_shape + (len(query_times),) + point_shape
    assert gs.allclose(result, expected)


@pytest.mark.parametrize("time", [0.25, gs.array(0.25)])
def test_tensor_valued_discrete_path_accepts_scalar_time(time):
    """Preserve all four point axes when a discrete path is evaluated once."""
    start = gs.reshape(gs.arange(16, dtype=gs.float64), (2, 2, 2, 2))
    data = gs.stack([start, start + 2.0])
    path = UniformlySampledDiscretePath(data, point_ndim=4)

    result = path(time)

    assert result.shape == (1, 2, 2, 2, 2)
    assert gs.allclose(result[0], start + 0.5)
