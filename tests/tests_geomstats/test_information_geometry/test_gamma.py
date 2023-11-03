import pytest

from geomstats.information_geometry.gamma import (
    GammaDistributions,
    GammaDistributionsRandomVariable,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import GammaRandomDataGenerator, RandomDataGenerator
from geomstats.test_cases.information_geometry.gamma import (
    GammaDistributionsTestCase,
    GammaMetricTestCase,
)

from .data.gamma import (
    GammaDistributionsSmokeTestData,
    GammaDistributionsTestData,
    GammaMetricTestData,
)


class TestGammaDistributions(
    GammaDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = GammaDistributions(equip=False)
    data_generator = GammaRandomDataGenerator(space)
    random_variable = GammaDistributionsRandomVariable(space)

    testing_data = GammaDistributionsTestData()


@pytest.mark.smoke
class TestGammaDistributionsSmoke(
    GammaDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = GammaDistributions(equip=False)
    testing_data = GammaDistributionsSmokeTestData()


class TestGammaMetric(GammaMetricTestCase, metaclass=DataBasedParametrizer):
    space = GammaDistributions()
    data_generator = RandomDataGenerator(space, amplitude=5.0)
    testing_data = GammaMetricTestData()
