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


class TestEuclidean(EuclideanTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(dim=random.randint(2, 5), equip=False)
    testing_data = EuclideanTestData()


class TestEuclideanMetric(EuclideanMetricTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(dim=random.randint(2, 5))
    testing_data = EuclideanMetricTestData()


@pytest.mark.smoke
class TestEuclideanMetric2(EuclideanMetricTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(dim=2)
    testing_data = EuclideanMetric2TestData()
