import random

import pytest

import geomstats.backend as gs
from geomstats.numerics.interpolation import UniformUnitIntervalLinearInterpolator
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import TestCase

from .data.interpolation import UniformUnitIntervalLinearInterpolatorTestData


@pytest.fixture(
    scope="class",
    params=[(1,), (2,)],
)
def interpolators(request):
    point_shape = request.param

    num_points = random.randint(5, 12)
    data = gs.random.uniform(size=(num_points,) + point_shape)

    request.cls.interpolator = UniformUnitIntervalLinearInterpolator(
        data, point_ndim=len(point_shape)
    )


@pytest.mark.usefixtures("interpolators")
class TestUniformUnitIntervalLinearInterpolator(
    TestCase, metaclass=DataBasedParametrizer
):
    testing_data = UniformUnitIntervalLinearInterpolatorTestData()

    def test_interpolate(self, t, expected, atol):
        res = self.interpolator(t)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_interpolate_uniformly(self, atol):
        times = gs.linspace(0.0, 1.0, num=self.interpolator._n_times)

        res = self.interpolator(times)
        self.assertAllClose(res, self.interpolator.data, atol=atol)
