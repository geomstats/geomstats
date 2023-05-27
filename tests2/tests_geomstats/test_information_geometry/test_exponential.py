from geomstats.information_geometry.exponential import (
    ExponentialDistributions,
    ExponentialMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.information_geometry.exponential import (
    ExponentialDistributionsTestCase,
    ExponentialMetricTestCase,
)
from tests2.tests_geomstats.test_information_geometry.data.exponential import (
    ExponentialDistributionsTestData,
    ExponentialMetricTestData,
)


class TestExponentialDistributions(
    ExponentialDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = ExponentialDistributions(equip=False)
    testing_data = ExponentialDistributionsTestData()


class TestExponentialMetric(ExponentialMetricTestCase, metaclass=DataBasedParametrizer):
    space = ExponentialDistributions(equip=False)
    space.equip_with_metric(ExponentialMetric)

    testing_data = ExponentialMetricTestData()
