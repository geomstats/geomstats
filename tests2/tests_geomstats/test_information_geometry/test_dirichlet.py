import random

import pytest

from geomstats.information_geometry.dirichlet import (
    DirichletDistributions,
    DirichletMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.information_geometry.dirichlet import (
    DirichletDistributionsTestCase,
    DirichletMetricTestCase,
)
from tests2.tests_geomstats.test_information_geometry.data.dirichlet import (
    DirichletDistributionsTestData,
    DirichletMetricTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = DirichletDistributions(dim=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestDirichletDistributions(
    DirichletDistributionsTestCase, metaclass=DataBasedParametrizer
):
    testing_data = DirichletDistributionsTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_spaces(request):
    space = request.cls.space = DirichletDistributions(dim=request.param, equip=False)
    space.equip_with_metric(DirichletMetric)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=5.0)


@pytest.mark.usefixtures("equipped_spaces")
class TestDirichletMetric(DirichletMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = DirichletMetricTestData()
