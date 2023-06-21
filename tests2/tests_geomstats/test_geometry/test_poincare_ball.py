import random

import pytest

from geomstats.geometry.poincare_ball import PoincareBall, PoincareBallMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.poincare_ball import (
    PoincareBallMetricTestCase,
    PoincareBallTestCase,
)

from .data.poincare_ball import (
    PoincareBall2MetricTestData,
    PoincareBall2TestData,
    PoincareBallMetricTestData,
    PoincareBallTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = PoincareBall(dim=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestPoincareBall(PoincareBallTestCase, metaclass=DataBasedParametrizer):
    testing_data = PoincareBallTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_spaces(request):
    request.cls.space = space = PoincareBall(dim=request.param, equip=False)
    space.equip_with_metric(PoincareBallMetric)


@pytest.mark.usefixtures("equipped_spaces")
class TestPoincareBallMetric(
    PoincareBallMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PoincareBallMetricTestData()


@pytest.mark.smoke
class TestPoincareBall2(PoincareBallTestCase, metaclass=DataBasedParametrizer):
    space = PoincareBall(dim=2, equip=False)
    testing_data = PoincareBall2TestData()


@pytest.mark.smoke
class TestPoincareBall2Metric(
    PoincareBallMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PoincareBall(dim=2, equip=True)
    testing_data = PoincareBall2MetricTestData()
