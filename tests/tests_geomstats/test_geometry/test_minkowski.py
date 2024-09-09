import random

import pytest

from geomstats.geometry.minkowski import Minkowski
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.euclidean import (
    EuclideanMetricTestCase,
    EuclideanTestCase,
)

from .data.minkowski import (
    Minkowski2MetricTestData,
    MinkowskiMetricTestData,
    MinkowskiTestData,
)


@pytest.mark.redundant
class TestMinkowski(EuclideanTestCase, metaclass=DataBasedParametrizer):
    space = Minkowski(dim=random.randint(2, 5), equip=False)

    testing_data = MinkowskiTestData()


class TestMinkowskiMetric(EuclideanMetricTestCase, metaclass=DataBasedParametrizer):
    space = Minkowski(dim=random.randint(2, 5))
    testing_data = MinkowskiMetricTestData()


@pytest.mark.smoke
class TestMinkowski2Metric(EuclideanMetricTestCase, metaclass=DataBasedParametrizer):
    space = Minkowski(dim=2)
    testing_data = Minkowski2MetricTestData()
