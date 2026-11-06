import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.siegel import Siegel, SiegelMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import assert_allclose
from geomstats.test_cases.geometry.base import ComplexVectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.siegel import SiegelMetricTestCase

from .data.base import ComplexVectorSpaceOpenSetTestData
from .data.siegel import (
    Siegel2MetricTestData,
    Siegel2TestData,
    Siegel3MetricTestData,
    SiegelMetricTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = Siegel(request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestSiegel(ComplexVectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = ComplexVectorSpaceOpenSetTestData()


@pytest.mark.smoke
class TestSiegel2(ComplexVectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = Siegel(2, equip=False)
    testing_data = Siegel2TestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_spaces(request):
    request.cls.space = Siegel(request.param)


@pytest.mark.usefixtures("equipped_spaces")
class TestSiegelMetric(SiegelMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = SiegelMetricTestData()


@pytest.mark.smoke
class TestSiegel2Metric(SiegelMetricTestCase, metaclass=DataBasedParametrizer):
    space = Siegel(2, equip=False)
    space.equip_with_metric(SiegelMetric)
    testing_data = Siegel2MetricTestData()


@pytest.mark.smoke
class TestSiegel3Metric(SiegelMetricTestCase, metaclass=DataBasedParametrizer):
    space = Siegel(3, equip=False)
    space.equip_with_metric(SiegelMetric)
    testing_data = Siegel3MetricTestData()


@pytest.mark.parametrize("dtype", [gs.complex64, gs.complex128])
@pytest.mark.parametrize("map_name", ["exp", "log"])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_rank_deficient_exp_log(symmetric, batched, map_name, dtype):
    """Rank-deficient directions retain their nonzero singular modes."""
    space = Siegel(2, symmetric=symmetric)
    direction = gs.array([[0.3j, 0j], [0j, 0j]], dtype=dtype)
    expected_point = gs.array([[gs.tanh(0.3) * 1j, 0j], [0j, 0j]], dtype=dtype)
    if batched:
        direction = gs.stack([direction, gs.zeros_like(direction)])
        expected_point = gs.stack([expected_point, gs.zeros_like(expected_point)])
    base_point = gs.zeros((2, 2), dtype=direction.dtype)

    assert gs.all(space.is_tangent(direction, base_point))
    assert gs.all(space.belongs(expected_point))
    if map_name == "exp":
        result = space.metric.exp(direction, base_point)
        assert_allclose(result, expected_point)
    else:
        result = space.metric.log(expected_point, base_point)
        assert_allclose(result, direction)
    assert result.dtype == direction.dtype
