import random

import pytest

from geomstats.information_geometry.categorical import (
    CategoricalDistributions,
    CategoricalMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.information_geometry.categorical import (
    CategoricalDistributionsTestCase,
    CategoricalMetricTestCase,
)
from tests2.tests_geomstats.test_information_geometry.data.categorical import (
    CategoricalDistributionsTestData,
    CategoricalMetricTestData,
)


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
    CategoricalDistributionsTestCase, metaclass=DataBasedParametrizer
):
    testing_data = CategoricalDistributionsTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_spaces(request):
    space = request.cls.space = CategoricalDistributions(dim=request.param, equip=False)
    space.equip_with_metric(CategoricalMetric)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)
    request.cls.data_generator_sphere = RandomDataGenerator(space.metric._sphere)


@pytest.mark.usefixtures("equipped_spaces")
class TestCategoricalMetric(CategoricalMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = CategoricalMetricTestData()
