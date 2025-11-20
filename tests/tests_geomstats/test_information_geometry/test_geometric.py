import pytest

from geomstats.information_geometry.geometric import (
    GeometricDistributions,
    GeometricDistributionsRandomVariable,
    GeometricMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.geometric import (
    GeometricDistributionsTestCase,
)

from .data.geometric import (
    GeometricDistributionsSmokeTestData,
    GeometricDistributionsTestData,
    GeometricMetricSmokeTestData,
    GeometricMetricTestData,
)


class TestGeometricDistributions(
    GeometricDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = GeometricDistributions(equip=False)
    random_variable = GeometricDistributionsRandomVariable(space)
    testing_data = GeometricDistributionsTestData()


@pytest.mark.smoke
class TestGeometricDistributionsSmoke(
    GeometricDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = GeometricDistributions(equip=False)
    testing_data = GeometricDistributionsSmokeTestData()


class TestGeometricMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = GeometricDistributions()
    data_generator = RandomDataGenerator(space, amplitude=10.0)
    testing_data = GeometricMetricTestData()


@pytest.mark.smoke
class TestGeometricMetricSmoke(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = GeometricDistributions(equip=False)
    space.equip_with_metric(GeometricMetric)
    testing_data = GeometricMetricSmokeTestData()
