import pytest

from geomstats.information_geometry.beta import (
    BetaDistributions,
    BetaDistributionsRandomVariable,
    BetaMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.information_geometry.beta import (
    BetaDistributionsTestCase,
    BetaMetricTestCase,
)

from .data.beta import (
    BetaDistributionsSmokeTestData,
    BetaDistributionsTestData,
    BetaMetricSmokeTestData,
    BetaMetricTestData,
)


class TestBetaDistributions(BetaDistributionsTestCase, metaclass=DataBasedParametrizer):
    space = BetaDistributions(equip=False)
    random_variable = BetaDistributionsRandomVariable(space)
    testing_data = BetaDistributionsTestData()


@pytest.mark.smoke
class TestBetaDistributionsSmoke(
    BetaDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = BetaDistributions(equip=False)
    testing_data = BetaDistributionsSmokeTestData()


@pytest.mark.slow
class TestBetaMetric(BetaMetricTestCase, metaclass=DataBasedParametrizer):
    space = BetaDistributions()

    data_generator = RandomDataGenerator(space, amplitude=5.0)

    testing_data = BetaMetricTestData()


@pytest.mark.smoke
class TestBetaMetricSmoke(BetaMetricTestCase, metaclass=DataBasedParametrizer):
    space = BetaDistributions(equip=False)
    space.equip_with_metric(BetaMetric)
    testing_data = BetaMetricSmokeTestData()
