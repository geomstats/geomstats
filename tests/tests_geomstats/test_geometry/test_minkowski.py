import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.minkowski import Minkowski
from geomstats.test_cases.geometry.euclidean import (
    EuclideanMetricTestCase,
    EuclideanTestCase,
)

from .data.euclidean import EuclideanTestData
from .data.minkowski import (
    Minkowski2MetricTestData,
    MinkowskiMetricTestData,
)


@pytest.mark.redundant
class TestMinkowski(EuclideanTestCase, metaclass=GeometricDataBasedParametrizer):
    space = Minkowski(dim=random.randint(2, 5), equip=False)

    testing_data = EuclideanTestData(space)


class TestMinkowskiMetric(
    EuclideanMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = Minkowski(dim=random.randint(2, 5))
    testing_data = MinkowskiMetricTestData(space)


@pytest.mark.smoke
class TestMinkowski2Metric(EuclideanMetricTestCase, metaclass=DataBasedParametrizer):
    space = Minkowski(dim=2)
    testing_data = Minkowski2MetricTestData()
