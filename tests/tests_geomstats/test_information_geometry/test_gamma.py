import pytest

from geomstats.information_geometry.gamma import (
    GammaDistributions,
    GammaDistributionsRandomVariable,
    NaturalToStandardDiffeo,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import GammaRandomDataGenerator, RandomDataGenerator
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.information_geometry.gamma import (
    GammaDistributionsTestCase,
    GammaMetricTestCase,
    StandardGammaDistributions,
)

from .data.gamma import (
    GammaDistributionsSmokeTestData,
    GammaDistributionsTestData,
    GammaMetricTestData,
    NaturalToStandardDiffeoTestData,
)


class TestNaturalToStandardDiffeo(DiffeoTestCase, metaclass=DataBasedParametrizer):
    space = GammaDistributions(equip=False)
    image_space = StandardGammaDistributions()
    diffeo = NaturalToStandardDiffeo()
    testing_data = NaturalToStandardDiffeoTestData()


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
    data_generator = RandomDataGenerator(space, amplitude=20.0)
    testing_data = GammaMetricTestData()
