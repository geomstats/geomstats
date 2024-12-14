import random

import pytest

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.information_geometry.multinomial import (
    MultinomialDistributions,
    SimplexToHypersphere,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import DiffeoBasedRandomDataGenerator, RandomDataGenerator
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.information_geometry.multinomial import (
    MultinomialDistributionsTestCase,
    MultinomialMetricTestCase,
)

from ..test_geometry.data.diffeo import DiffeoTestData
from .data.multinomial import (
    MultinomialDistributions2TestData,
    MultinomialDistributions3TestData,
    MultinomialDistributionsTestData,
    MultinomialMetricTestData,
)


class TestSimplexToHypersphere(DiffeoTestCase, metaclass=DataBasedParametrizer):
    dim = random.randint(2, 5)
    space = MultinomialDistributions(dim=dim, n_draws=1, equip=False)
    image_space = Hypersphere(dim=dim, equip=False)
    diffeo = SimplexToHypersphere()

    image_data_generator = DiffeoBasedRandomDataGenerator(space, diffeo)

    testing_data = DiffeoTestData()


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


@pytest.mark.smoke
class TestMultinomialDistributions2(
    MultinomialDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = MultinomialDistributions(dim=2, n_draws=3, equip=False)
    testing_data = MultinomialDistributions2TestData()


@pytest.mark.smoke
class TestMultinomialDistributions3(
    MultinomialDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = MultinomialDistributions(dim=3, n_draws=3, equip=False)
    testing_data = MultinomialDistributions3TestData()


@pytest.fixture(
    scope="class",
    params=[
        (2, 5),
        (random.randint(3, 5), random.randint(2, 10)),
    ],
)
def equipped_spaces(request):
    dim, n_draws = request.param
    space = request.cls.space = MultinomialDistributions(dim=dim, n_draws=n_draws)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)
    request.cls.data_generator_sphere = RandomDataGenerator(space.metric._sphere)


@pytest.mark.usefixtures("equipped_spaces")
class TestMultinomialMetric(MultinomialMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = MultinomialMetricTestData()
