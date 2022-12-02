import random

import pytest

from geomstats.geometry.stiefel import Stiefel
from geomstats.test.geometry.stiefel import (
    StiefelStaticMethodsTestCase,
    StiefelTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer, TestBasedParametrizer
from tests2.data.stiefel_data import StiefelStaticMethodsTestData, StiefelTestData


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
    request.cls.space = Stiefel(n=n, p=p)


@pytest.mark.usefixtures("spaces")
class TestStiefel(StiefelTestCase, metaclass=DataBasedParametrizer):
    testing_data = StiefelTestData()


class TestStiefelStaticMethods(
    StiefelStaticMethodsTestCase, metaclass=TestBasedParametrizer
):
    Space = Stiefel
    testing_data = StiefelStaticMethodsTestData()
