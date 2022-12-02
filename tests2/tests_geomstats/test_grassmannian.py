import random

import pytest

from geomstats.geometry.grassmannian import Grassmannian
from geomstats.test.geometry.grassmannian import GrassmannianTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.grassmannian_data import Grassmannian32TestData, GrassmannianTestData


def _get_random_params():

    while True:
        a = random.randint(2, 6)
        b = random.randint(2, 6)

        if a != b:
            break

    if a > b:
        n, p = a, b
    else:
        n, p = b, a

    return n, p


@pytest.fixture(
    scope="class",
    params=[(3, 2), _get_random_params()],
)
def spaces(request):
    n, p = request.param
    request.cls.space = Grassmannian(n=n, p=p)


@pytest.mark.usefixtures("spaces")
class TestGrassmannian(GrassmannianTestCase, metaclass=DataBasedParametrizer):
    testing_data = GrassmannianTestData()


class TestGrassmannian32(GrassmannianTestCase, metaclass=DataBasedParametrizer):
    space = Grassmannian(3, 2)
    testing_data = Grassmannian32TestData()
