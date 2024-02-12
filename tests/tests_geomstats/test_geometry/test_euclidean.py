import random

import pytest

from geomstats.geometry.euclidean import Euclidean, FlatRiemannianMetric
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.euclidean import (
    EuclideanMetricTestCase,
    EuclideanTestCase,
    FlatRiemannianMetricTestCase,
)

from .data.euclidean import (
    EuclideanMetric2TestData,
    EuclideanMetricTestData,
    EuclideanTestData,
    FlatRiemannianMetricTestData,
)


class TestEuclidean(EuclideanTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(dim=random.randint(2, 5), equip=False)
    testing_data = EuclideanTestData()


class TestFlatRiemannianMetric(
    FlatRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    _dim = random.randint(2, 5)
    space = Euclidean(_dim, equip=False).equip_with_metric(
        FlatRiemannianMetric,
        metric_matrix=SPDMatrices(_dim, equip=False).random_point(),
    )
    testing_data = FlatRiemannianMetricTestData()


class TestEuclideanMetric(EuclideanMetricTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(dim=random.randint(2, 5))
    testing_data = EuclideanMetricTestData()


@pytest.mark.smoke
class TestEuclideanMetric2(EuclideanMetricTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(dim=2)
    testing_data = EuclideanMetric2TestData()
