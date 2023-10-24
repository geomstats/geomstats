import random

import pytest

from geomstats.geometry.full_rank_matrices import FullRankMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase

from .data.base import OpenSetTestData
from .data.full_rank_matrices import FullRankMatrices32TestData
from .data.matrices import MatricesMetricTestData


@pytest.fixture(
    scope="class",
    params=[
        (3, 3),
        (random.randint(3, 5), random.randint(3, 5)),
    ],
)
def spaces(request):
    n, k = request.param
    request.cls.space = FullRankMatrices(n, k, equip=False)


@pytest.mark.usefixtures("spaces")
class TestFullRankMatrices(OpenSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = OpenSetTestData()


@pytest.mark.smoke
class TestFullRankMatrices32(OpenSetTestCase, metaclass=DataBasedParametrizer):
    space = FullRankMatrices(3, 2, equip=False)
    testing_data = FullRankMatrices32TestData()


@pytest.mark.redundant
class TestMatricesMetric(MatricesMetricTestCase, metaclass=DataBasedParametrizer):
    n, k = random.randint(3, 5), random.randint(3, 5)
    space = FullRankMatrices(n, k)
    testing_data = MatricesMetricTestData()
