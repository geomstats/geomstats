import random

import pytest

from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.matrices import (
    MatricesMetricTestCase,
    MatricesTestCase,
)

from .data.matrices import MatricesMetricTestData, MatricesTestData


@pytest.fixture(
    scope="class",
    params=[
        (2, 3),
        (random.randint(3, 5), random.randint(3, 5)),
    ],
)
def spaces(request):
    m, n = request.param
    request.cls.space = Matrices(m=m, n=n, equip=False)


@pytest.mark.usefixtures("spaces")
class TestMatrices(MatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = MatricesTestData()


@pytest.fixture(
    scope="class",
    params=[
        (2, 3),
        (random.randint(3, 5), random.randint(3, 5)),
    ],
)
def equipped_spaces(request):
    m, n = request.param
    space = Matrices(m=m, n=n, equip=False)
    request.cls.space = space
    space.equip_with_metric(MatricesMetric)


@pytest.mark.usefixtures("equipped_spaces")
class TestMatricesMetric(MatricesMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = MatricesMetricTestData()
