import random

import pytest

from geomstats.geometry.rank_k_psd_matrices import RankKPSDMatrices
from geomstats.test.geometry.rank_k_psd_matrices import RankKPSDMatricesTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.rank_k_psd_matrices_data import RankKPSDMatricesTestData


def _get_random_params():
    while True:
        a = random.randint(2, 6)
        b = random.randint(2, 6)

        if a != b:
            break

    if a > b:
        n, k = a, b
    else:
        n, k = b, a

    return n, k


@pytest.fixture(
    scope="class",
    params=[
        (3, 2),
        _get_random_params(),
    ],
)
def spaces(request):
    n, k = request.param
    request.cls.space = RankKPSDMatrices(n=n, k=k)


@pytest.mark.usefixtures("spaces")
class TestRankKPSDMatrices(RankKPSDMatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = RankKPSDMatricesTestData()
