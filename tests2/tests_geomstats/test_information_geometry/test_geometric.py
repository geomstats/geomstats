from geomstats.information_geometry.geometric import (
    GeometricDistributions,
    GeometricMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.information_geometry.geometric import (
    GeometricDistributionsTestCase,
    GeometricMetricTestCase,
)
from tests2.tests_geomstats.test_information_geometry.data.geometric import (
    GeometricDistributionsTestData,
    GeometricMetricTestData,
)


class TestGeometricDistributions(
    GeometricDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = GeometricDistributions(equip=False)
    testing_data = GeometricDistributionsTestData()


class TestGeometricMetric(GeometricMetricTestCase, metaclass=DataBasedParametrizer):
    space = GeometricDistributions(equip=False)
    space.equip_with_metric(GeometricMetric)

    data_generator = RandomDataGenerator(space, amplitude=10.0)

    testing_data = GeometricMetricTestData()
