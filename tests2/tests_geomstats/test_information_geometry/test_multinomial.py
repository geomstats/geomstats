import random

import pytest

from geomstats.information_geometry.multinomial import (
    MultinomialDistributions,
    MultinomialMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.information_geometry.multinomial import (
    MultinomialDistributionsTestCase,
    MultinomialMetricTestCase,
)
from tests2.tests_geomstats.test_information_geometry.data.multinomial import (
    MultinomialDistributionsTestData,
    MultinomialMetricTestData,
)


@pytest.fixture(
    scope="class",
    params=[(2, 5), (random.randint(3, 5), random.randint(2, 10))],
)
def spaces(request):
    dim, n_draws = request.param
    request.cls.space = MultinomialDistributions(dim=dim, n_draws=n_draws, equip=False)


@pytest.mark.usefixtures("spaces")
class TestMultinomialDistributions(
    MultinomialDistributionsTestCase, metaclass=DataBasedParametrizer
):
    testing_data = MultinomialDistributionsTestData()


@pytest.fixture(
    scope="class",
    params=[
        (2, 5),
        (random.randint(3, 5), random.randint(2, 10)),
    ],
)
def equipped_spaces(request):
    dim, n_draws = request.param
    space = request.cls.space = MultinomialDistributions(
        dim=dim, n_draws=n_draws, equip=False
    )
    space.equip_with_metric(MultinomialMetric)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)
    request.cls.data_generator_sphere = RandomDataGenerator(space.metric._sphere)


@pytest.mark.usefixtures("equipped_spaces")
class TestMultinomialMetric(MultinomialMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = MultinomialMetricTestData()
