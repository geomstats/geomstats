import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.information_geometry.multinomial import MultinomialDistributions
from geomstats.test_cases.information_geometry.multinomial import (
    MultinomialDistributionsTestCase,
    MultinomialMetricTestCase,
)

from .data.multinomial import (
    MultinomialDistributions2TestData,
    MultinomialDistributions3TestData,
    MultinomialDistributionsTestData,
    MultinomialMetricTestData,
)


class TestMultinomialDistributions(
    MultinomialDistributionsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = MultinomialDistributions(
        dim=random.randint(2, 5), n_draws=random.randint(2, 10), equip=False
    )
    testing_data = MultinomialDistributionsTestData(space)


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


class TestMultinomialMetric(
    MultinomialMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = MultinomialDistributions(
        dim=random.randint(2, 5), n_draws=random.randint(2, 10)
    )
    testing_data = MultinomialMetricTestData(
        space,
        generator_kwargs={"within_injectivity_radius": True},
    )
