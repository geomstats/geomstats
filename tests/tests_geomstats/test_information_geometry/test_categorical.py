import random

import pytest

from geomstats.information_geometry.categorical import CategoricalDistributions
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.information_geometry.multinomial import (
    MultinomialDistributionsTestCase,
    MultinomialMetricTestCase,
)

from .data.categorical import CategoricalMetricTestData
from .data.multinomial import MultinomialDistributionsTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = CategoricalDistributions(dim=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestCategoricalDistributions(
    MultinomialDistributionsTestCase, metaclass=DataBasedParametrizer
):
    testing_data = MultinomialDistributionsTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_spaces(request):
    space = request.cls.space = CategoricalDistributions(dim=request.param)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)
    request.cls.data_generator_sphere = RandomDataGenerator(space.metric._sphere)


@pytest.mark.redundant
@pytest.mark.usefixtures("equipped_spaces")
class TestCategoricalMetric(MultinomialMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = CategoricalMetricTestData()
