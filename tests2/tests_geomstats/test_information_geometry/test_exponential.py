import pytest

from geomstats.information_geometry.exponential import (
    ExponentialDistributions,
    ExponentialDistributionsRandomVariable,
    ExponentialMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.exponential import (
    ExponentialDistributionsTestCase,
)

from .data.exponential import (
    ExponentialDistributionsSmokeTestData,
    ExponentialDistributionsTestData,
    ExponentialMetricSmokeTestData,
    ExponentialMetricTestData,
)


class TestExponentialDistributions(
    ExponentialDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = ExponentialDistributions(equip=False)
    random_variable = ExponentialDistributionsRandomVariable(space)
    testing_data = ExponentialDistributionsTestData()


@pytest.mark.smoke
class TestExponentialDistributionsSmoke(
    ExponentialDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = ExponentialDistributions(equip=False)
    testing_data = ExponentialDistributionsSmokeTestData()


class TestExponentialMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = ExponentialDistributions()
    testing_data = ExponentialMetricTestData()


@pytest.mark.smoke
class TestExponentialMetricSmoke(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = ExponentialDistributions(equip=False)
    space.equip_with_metric(ExponentialMetric)
    testing_data = ExponentialMetricSmokeTestData()
