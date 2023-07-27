import random

import pytest

from geomstats.geometry.minkowski import Minkowski
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.euclidean import EuclideanTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.minkowski import (
    Minkowski2MetricTestData,
    MinkowskiMetricTestData,
    MinkowskiTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = Minkowski(dim=request.param)


@pytest.mark.usefixtures("spaces")
class TestMinkowski(EuclideanTestCase, metaclass=DataBasedParametrizer):
    # bare minimum testing as Euclidean covers all cases
    testing_data = MinkowskiTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_spaces(request):
    request.cls.space = Minkowski(dim=request.param)


@pytest.mark.usefixtures("equipped_spaces")
class TestMinkowskiMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = MinkowskiMetricTestData()


@pytest.mark.smoke
class TestMinkowski2Metric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = Minkowski(dim=2)
    testing_data = Minkowski2MetricTestData()
