import random

import pytest

from geomstats.geometry.stiefel import Stiefel, StiefelCanonicalMetric
from geomstats.test.geometry.stiefel import (
    StiefelCanonicalMetricTestCase,
    StiefelStaticMethodsTestCase,
    StiefelTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer, Parametrizer
from tests2.data.stiefel_data import (
    StiefelCanonicalMetricTestData,
    StiefelStaticMethodsTestData,
    StiefelTestData,
)


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
    request.cls.space = Stiefel(n=n, p=p, equip=False)


@pytest.mark.usefixtures("spaces")
class TestStiefel(StiefelTestCase, metaclass=DataBasedParametrizer):
    testing_data = StiefelTestData()


class TestStiefelStaticMethods(StiefelStaticMethodsTestCase, metaclass=Parametrizer):
    Space = Stiefel
    testing_data = StiefelStaticMethodsTestData()


@pytest.fixture(
    scope="class",
    params=[(3, 2), _get_random_params()],
)
def equipped_spaces(request):
    n, p = request.param
    space = Stiefel(n=n, p=p, equip=False)
    request.cls.space = space
    space.equip_with_metric(StiefelCanonicalMetric)


@pytest.mark.usefixtures("equipped_spaces")
class TestStiefelCanonicalMetric(
    StiefelCanonicalMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = StiefelCanonicalMetricTestData()
