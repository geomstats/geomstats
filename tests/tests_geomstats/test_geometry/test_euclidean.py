import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import (
    CoordinateScalingDiffeo,
    Euclidean,
    EuclideanMetric,
)
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.euclidean import (
    CanonicalEuclideanMetricTestCase,
    EuclideanMetricTestCase,
    EuclideanTestCase,
)

from .data.diffeo import DiffeoTestData
from .data.euclidean import (
    CanonicalEuclideanMetric2TestData,
    CanonicalEuclideanMetricTestData,
    EuclideanMetricTestData,
    EuclideanTestData,
)


class TestCoordinateScalingDiffeo(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _dim = random.randint(2, 5)
    space = Euclidean(dim=_dim, equip=False)
    image_space = Euclidean(dim=_dim, equip=False)
    diffeo = CoordinateScalingDiffeo(gs.random.uniform(0.5, 2.0, size=(_dim,)))
    testing_data = DiffeoTestData()


class TestEuclidean(EuclideanTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(dim=random.randint(2, 5), equip=False)
    testing_data = EuclideanTestData()


class TestEuclideanMetric(EuclideanMetricTestCase, metaclass=DataBasedParametrizer):
    _dim = random.randint(2, 5)
    space = Euclidean(_dim, equip=False).equip_with_metric(
        EuclideanMetric,
        metric_matrix=SPDMatrices(_dim, equip=False).random_point(),
    )
    testing_data = EuclideanMetricTestData()


class TestCanonicalEuclideanMetric(
    CanonicalEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    space = Euclidean(dim=random.randint(2, 5))
    testing_data = CanonicalEuclideanMetricTestData()


@pytest.mark.smoke
class TestCanonicalEuclideanMetric2(
    CanonicalEuclideanMetricTestCase, metaclass=DataBasedParametrizer
):
    space = Euclidean(dim=2)
    testing_data = CanonicalEuclideanMetric2TestData()
