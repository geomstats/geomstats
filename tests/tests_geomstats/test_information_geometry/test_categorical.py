import random

import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.categorical import CategoricalDistributions
from geomstats.test_cases.information_geometry.multinomial import (
    MultinomialDistributionsTestCase,
    MultinomialMetricTestCase,
)

from .data.multinomial import (
    MultinomialDistributionsTestData,
    MultinomialMetricTestData,
)


class TestCategoricalDistributions(
    MultinomialDistributionsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = CategoricalDistributions(dim=random.randint(2, 5), equip=False)
    testing_data = MultinomialDistributionsTestData(space)


@pytest.mark.redundant
class TestCategoricalMetric(
    MultinomialMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = CategoricalDistributions(dim=random.randint(2, 5))
    testing_data = MultinomialMetricTestData(
        space,
        method_marks={"xfail": ("log_after_exp",)},
        generator_kwargs={"amplitude": 10.0},
    )
