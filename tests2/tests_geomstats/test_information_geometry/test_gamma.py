from geomstats.information_geometry.gamma import GammaDistributions, GammaMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import GammaRandomDataGenerator, RandomDataGenerator
from geomstats.test_cases.information_geometry.gamma import (
    GammaDistributionsTestCase,
    GammaMetricTestCase,
)
from tests2.tests_geomstats.test_information_geometry.data.gamma import (
    GammaDistributionsTestData,
    GammaMetricTestData,
)


class TestGammaDistributions(
    GammaDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = GammaDistributions(equip=False)
    data_generator = GammaRandomDataGenerator(space)

    testing_data = GammaDistributionsTestData()


class TestGammaMetric(GammaMetricTestCase, metaclass=DataBasedParametrizer):
    space = GammaDistributions(equip=False)
    space.equip_with_metric(GammaMetric)
    data_generator = RandomDataGenerator(space, amplitude=5.0)

    testing_data = GammaMetricTestData()
