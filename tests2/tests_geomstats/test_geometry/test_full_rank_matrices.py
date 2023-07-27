import random

import pytest

from geomstats.geometry.full_rank_matrices import FullRankMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

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


@pytest.fixture(
    scope="class",
    params=[
        (3, 3),
        (random.randint(3, 5), random.randint(3, 5)),
    ],
)
def equipped_spaces(request):
    n, k = request.param
    request.cls.space = FullRankMatrices(n, k, equip=True)


@pytest.mark.usefixtures("equipped_spaces")
class TestMatricesMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = MatricesMetricTestData()
