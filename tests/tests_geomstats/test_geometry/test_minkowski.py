import random

import pytest

from geomstats.geometry.minkowski import Minkowski
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.euclidean import EuclideanTestCase
from geomstats.test_cases.geometry.flat_riemannian_metric import (
    FlatRiemannianMetricTestCase,
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


class TestMinkowskiMetric(
    FlatRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = Minkowski(dim=random.randint(2, 5))
    testing_data = MinkowskiMetricTestData()


@pytest.mark.smoke
class TestMinkowski2Metric(
    FlatRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = Minkowski(dim=2)
    testing_data = Minkowski2MetricTestData()
