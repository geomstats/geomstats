import random

import pytest

from geomstats.geometry.hermitian import Hermitian
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.hermitian import (
    HermitianMetricTestCase,
    HermitianTestCase,
)

from .data.hermitian import (
    Hermitian2MetricTestData,
    HermitianMetricTestData,
    HermitianTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = Hermitian(dim=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestHermitian(HermitianTestCase, metaclass=DataBasedParametrizer):
    testing_data = HermitianTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_spaces(request):
    request.cls.space = Hermitian(dim=request.param)


@pytest.mark.usefixtures("equipped_spaces")
class TestHermitianMetric(HermitianMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = HermitianMetricTestData()


@pytest.mark.smoke
class TestHermitian2Metric(HermitianMetricTestCase, metaclass=DataBasedParametrizer):
    space = Hermitian(dim=2, equip=True)
    testing_data = Hermitian2MetricTestData()
