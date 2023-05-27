from geomstats.information_geometry.poisson import PoissonDistributions, PoissonMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.information_geometry.poisson import (
    PoissonDistributionsTestCase,
    PoissonMetricTestCase,
)
from tests2.tests_geomstats.test_information_geometry.data.poisson import (
    PoissonDistributionsTestData,
    PoissonMetricTestData,
)


class TestPoissonDistributions(
    PoissonDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = PoissonDistributions(equip=False)
    testing_data = PoissonDistributionsTestData()


class TestPoissonMetric(PoissonMetricTestCase, metaclass=DataBasedParametrizer):
    space = PoissonDistributions(equip=False)
    space.equip_with_metric(PoissonMetric)

    data_generator = RandomDataGenerator(space, amplitude=5.0)

    testing_data = PoissonMetricTestData()
