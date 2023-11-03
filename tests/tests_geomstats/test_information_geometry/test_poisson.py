import pytest

from geomstats.information_geometry.poisson import (
    PoissonDistributions,
    PoissonDistributionsRandomVariable,
    PoissonMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.poisson import (
    PoissonDistributionsTestCase,
)

from .data.poisson import (
    PoissonDistributionsSmokeTestData,
    PoissonDistributionsTestData,
    PoissonMetricSmokeTestData,
    PoissonMetricTestData,
)


class TestPoissonDistributions(
    PoissonDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = PoissonDistributions(equip=False)
    random_variable = PoissonDistributionsRandomVariable(space)
    testing_data = PoissonDistributionsTestData()


@pytest.mark.smoke
class TestPoissonDistributionsSmoke(
    PoissonDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = PoissonDistributions(equip=False)
    testing_data = PoissonDistributionsSmokeTestData()


class TestPoissonMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = PoissonDistributions()
    data_generator = RandomDataGenerator(space, amplitude=5.0)
    testing_data = PoissonMetricTestData()


@pytest.mark.smoke
class TestPoissonMetricSmoke(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = PoissonDistributions(equip=False)
    space.equip_with_metric(PoissonMetric)
    testing_data = PoissonMetricSmokeTestData()
