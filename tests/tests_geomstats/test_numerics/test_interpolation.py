import random

import pytest

import geomstats.backend as gs
from geomstats.numerics.interpolation import (
    LinearInterpolator1D,
    UniformUnitIntervalLinearInterpolator,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.numerics.interpolation import InterpolatorTestCase

from .data.interpolation import InterpolatorTestData


@pytest.fixture(
    scope="class",
    params=[((1,), True), ((2,), True), ((1,), False), ((2,), False)],
)
def interpolators(request):
    point_shape, uniform = request.param

    num_points = random.randint(5, 12)

    data = request.cls.data = gs.random.uniform(size=(num_points,) + point_shape)

    if uniform:
        request.cls.times = gs.linspace(0.0, 1.0, num=num_points)
        request.cls.interpolator = UniformUnitIntervalLinearInterpolator(
            data, point_ndim=len(point_shape)
        )
    else:
        times = request.cls.times = gs.sort(
            gs.random.uniform(0.0, 1.0, size=(num_points,))
        )
        request.cls.interpolator = LinearInterpolator1D(
            times, data, point_ndim=len(point_shape)
        )


@pytest.mark.usefixtures("interpolators")
class TestLinearInterpolator(InterpolatorTestCase, metaclass=DataBasedParametrizer):
    testing_data = InterpolatorTestData()
