import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.information_geometry.multinomial import (
    MultinomialDistributions,
    SimplexToPositiveHypersphere,
)
from geomstats.test.random import DiffeoBasedRandomDataGenerator
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


class TestSimplexToPositiveHypersphere(
    DiffeoTestCase, metaclass=GeometricDataBasedParametrizer
):
    _dim = random.randint(2, 5)
    domain_space = MultinomialDistributions(dim=_dim, n_draws=1, equip=False)
    image_space = Hypersphere(dim=_dim, equip=False)
    diffeo = SimplexToPositiveHypersphere()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )
    # NB: image points must lie in the positive orthant
    testing_data.image_space_data_generator = DiffeoBasedRandomDataGenerator(
        domain_space, diffeo
    )


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


class TestMultinomialDistributions(
    MultinomialDistributionsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = MultinomialDistributions(
        dim=random.randint(2, 5), n_draws=random.randint(2, 10), equip=False
    )
    testing_data = MultinomialDistributionsTestData(space)


class TestMultinomialMetric(
    MultinomialMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = MultinomialDistributions(
        dim=random.randint(2, 5), n_draws=random.randint(2, 10)
    )
    testing_data = MultinomialMetricTestData(
        space, method_marks={"xfail": ("log_after_exp",)}
    )
