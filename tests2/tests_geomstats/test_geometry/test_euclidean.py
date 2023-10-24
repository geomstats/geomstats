import random

import pytest

from geomstats.geometry.euclidean import Euclidean
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.euclidean import (
    EuclideanMetricTestCase,
    EuclideanTestCase,
)

from .data.euclidean import (
    EuclideanMetric2TestData,
    EuclideanMetricTestData,
    EuclideanTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = Euclidean(dim=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestEuclidean(EuclideanTestCase, metaclass=DataBasedParametrizer):
    testing_data = EuclideanTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_spaces(request):
    request.cls.space = Euclidean(dim=request.param)


@pytest.mark.usefixtures("equipped_spaces")
class TestEuclideanMetric(EuclideanMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = EuclideanMetricTestData()


@pytest.mark.smoke
class TestEuclideanMetric2(EuclideanMetricTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(dim=2)
    testing_data = EuclideanMetric2TestData()
