from geomstats.information_geometry.beta import BetaDistributions, BetaMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)
from geomstats.test_cases.information_geometry.dirichlet import DirichletMetricTestCase
from tests2.tests_geomstats.test_information_geometry.data.beta import (
    BetaDistributionsTestData,
    BetaMetricTestData,
)


class TestBetaDistributions(
    InformationManifoldMixinTestCase, OpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = BetaDistributions(equip=False)
    testing_data = BetaDistributionsTestData()


class TestBetaMetric(DirichletMetricTestCase, metaclass=DataBasedParametrizer):
    space = BetaDistributions(equip=False)
    space.equip_with_metric(BetaMetric)

    data_generator = RandomDataGenerator(space, amplitude=5.0)

    testing_data = BetaMetricTestData()
