import random

import pytest

from geomstats.geometry.full_rank_matrices import FullRankMatrices
from geomstats.test.geometry.full_rank_matrices import FullRankMatricesTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.full_rank_matrices_data import FullRankMatricesTestData


@pytest.fixture(
    scope="class",
    params=[
        (3, 3),
        (random.randint(3, 5), random.randint(3, 5)),
    ],
)
def spaces(request):
    n, k = request.param
    request.cls.space = FullRankMatrices(n, k)


@pytest.mark.usefixtures("spaces")
class TestFullRankMatrices(FullRankMatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = FullRankMatricesTestData()
