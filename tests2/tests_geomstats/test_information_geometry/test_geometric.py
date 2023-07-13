from geomstats.information_geometry.geometric import (
    GeometricDistributions,
    GeometricMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)
from tests2.tests_geomstats.test_information_geometry.data.geometric import (
    GeometricDistributionsTestData,
    GeometricMetricTestData,
)


class TestGeometricDistributions(
    InformationManifoldMixinTestCase, OpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = GeometricDistributions(equip=False)
    testing_data = GeometricDistributionsTestData()


class TestGeometricMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = GeometricDistributions(equip=False)
    space.equip_with_metric(GeometricMetric)

    data_generator = RandomDataGenerator(space, amplitude=10.0)

    testing_data = GeometricMetricTestData()
