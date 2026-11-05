import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.positive_reals import PositiveReals
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
    RiemannianMetricTestCase,
)
from geomstats.variants.geometry.positive_reals import (
    LogDiffeo,
    PullbackPositiveRealsMetric,
)

from .data.positive_reals import (
    LogDiffeoTestData,
    PositiveRealsMetricTestData,
    PositiveRealsTestData,
)
from .data.riemannian_metric import RiemannianMetricComparisonTestData


class TestPositiveReals(
    VectorSpaceOpenSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveReals(equip=False)
    testing_data = PositiveRealsTestData(space)


class TestPositiveRealsMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveReals()
    testing_data = PositiveRealsMetricTestData(space)


class TestLogDiffeo(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    domain_space = PositiveReals(equip=False)
    image_space = Euclidean(dim=1, equip=False)
    diffeo = LogDiffeo()

    testing_data = LogDiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


@pytest.mark.redundant
class TestPullbackPositiveRealsMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveReals(equip=False).equip_with_metric(PullbackPositiveRealsMetric)
    testing_data = PositiveRealsMetricTestData(space)


class TestPullbackPositiveRealsMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveReals()
    other_space = PositiveReals(equip=False).equip_with_metric(
        PullbackPositiveRealsMetric
    )
    testing_data = RiemannianMetricComparisonTestData(space)
