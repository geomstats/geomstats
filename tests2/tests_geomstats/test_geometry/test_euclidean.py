import random

import pytest

from geomstats.geometry.euclidean import Euclidean
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.euclidean import (
    EuclideanMetricTestCase,
    EuclideanTestCase,
)

from .data.euclidean import EuclideanMetricTestData, EuclideanTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = Euclidean(dim=request.param)


@pytest.mark.usefixtures("spaces")
class TestEuclidean(EuclideanTestCase, metaclass=DataBasedParametrizer):
    testing_data = EuclideanTestData()


@pytest.mark.usefixtures("spaces")
class TestEuclideanMetric(EuclideanMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = EuclideanMetricTestData()
