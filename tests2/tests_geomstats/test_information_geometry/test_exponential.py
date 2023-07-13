from geomstats.information_geometry.exponential import (
    ExponentialDistributions,
    ExponentialMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)
from tests2.tests_geomstats.test_information_geometry.data.exponential import (
    ExponentialDistributionsTestData,
    ExponentialMetricTestData,
)


class TestExponentialDistributions(
    InformationManifoldMixinTestCase, OpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = ExponentialDistributions(equip=False)
    testing_data = ExponentialDistributionsTestData()


class TestExponentialMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = ExponentialDistributions(equip=False)
    space.equip_with_metric(ExponentialMetric)

    testing_data = ExponentialMetricTestData()
