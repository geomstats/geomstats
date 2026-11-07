import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test_cases.geometry.euclidean import (
    CanonicalEuclideanMetricTestCase,
    EuclideanMetricTestCase,
    EuclideanTestCase,
)

from .data.euclidean import (
    CanonicalEuclideanMetric2TestData,
    CanonicalEuclideanMetricTestData,
    EuclideanMetricTestData,
    EuclideanTestData,
)


class TestEuclidean(EuclideanTestCase, metaclass=GeometricDataBasedParametrizer):
    space = Euclidean(dim=random.randint(2, 5), equip=False)
    testing_data = EuclideanTestData(space)


class TestEuclideanMetric(
    EuclideanMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _dim = random.randint(2, 5)
    space = Euclidean(_dim, equip=False).equip_with_metric(
        EuclideanMetric,
        metric_matrix=SPDMatrices(_dim, equip=False).random_point(),
    )
    testing_data = EuclideanMetricTestData(space)


class TestCanonicalEuclideanMetric(
    CanonicalEuclideanMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = Euclidean(dim=random.randint(2, 5))
    testing_data = CanonicalEuclideanMetricTestData(space)


@pytest.mark.smoke
class TestCanonicalEuclideanMetric2(
    CanonicalEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    space = Euclidean(dim=2)
    testing_data = CanonicalEuclideanMetric2TestData()
