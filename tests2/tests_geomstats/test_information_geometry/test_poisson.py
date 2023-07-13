from geomstats.information_geometry.poisson import PoissonDistributions, PoissonMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)
from tests2.tests_geomstats.test_information_geometry.data.poisson import (
    PoissonDistributionsTestData,
    PoissonMetricTestData,
)


class TestPoissonDistributions(
    InformationManifoldMixinTestCase, OpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = PoissonDistributions(equip=False)
    testing_data = PoissonDistributionsTestData()


class TestPoissonMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = PoissonDistributions(equip=False)
    space.equip_with_metric(PoissonMetric)

    data_generator = RandomDataGenerator(space, amplitude=5.0)

    testing_data = PoissonMetricTestData()
