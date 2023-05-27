from geomstats.information_geometry.beta import BetaDistributions, BetaMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.information_geometry.beta import (
    BetaDistributionsTestCase,
    BetaMetricTestCase,
)
from tests2.tests_geomstats.test_information_geometry.data.beta import (
    BetaDistributionsTestData,
    BetaMetricTestData,
)


class TestBetaDistributions(BetaDistributionsTestCase, metaclass=DataBasedParametrizer):
    space = BetaDistributions(equip=False)
    testing_data = BetaDistributionsTestData()


class TestBetaMetric(BetaMetricTestCase, metaclass=DataBasedParametrizer):
    space = BetaDistributions(equip=False)
    space.equip_with_metric(BetaMetric)

    data_generator = RandomDataGenerator(space, amplitude=5.0)

    testing_data = BetaMetricTestData()
