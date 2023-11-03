import random

import pytest

from geomstats.information_geometry.dirichlet import DirichletDistributions
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.information_geometry.dirichlet import (
    DirichletDistributionsTestCase,
    DirichletMetricTestCase,
)

from .data.dirichlet import (
    DirichletDistributions3TestData,
    DirichletDistributionsTestData,
    DirichletMetricTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        random.randint(3, 4),
    ],
)
def spaces(request):
    request.cls.space = DirichletDistributions(dim=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestDirichletDistributions(
    DirichletDistributionsTestCase, metaclass=DataBasedParametrizer
):
    testing_data = DirichletDistributionsTestData()


@pytest.mark.smoke
class TestDirichletDistributions3(
    DirichletDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = DirichletDistributions(dim=3, equip=False)
    testing_data = DirichletDistributions3TestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(3, 4),
    ],
)
def equipped_spaces(request):
    space = request.cls.space = DirichletDistributions(dim=request.param)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=5.0)


@pytest.mark.usefixtures("equipped_spaces")
@pytest.mark.slow
class TestDirichletMetric(DirichletMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = DirichletMetricTestData()
