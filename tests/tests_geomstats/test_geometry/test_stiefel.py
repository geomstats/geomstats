import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.stiefel import Stiefel, StiefelCanonicalMetric
from geomstats.test_cases.geometry.stiefel import (
    StiefelCanonicalMetricTestCase,
    StiefelStaticMethodsTestCase,
    StiefelTestCase,
)

from .data.stiefel import (
    StiefelCanonicalMetricSquareTestData,
    StiefelCanonicalMetricTestData,
    StiefelStaticMethodsTestData,
    StiefelTestData,
)

# TODO: use comparison with limit cases
# TODO: make it work with p=1?


def _get_random_params():
    while True:
        a = random.randint(2, 5)
        b = random.randint(2, 5)

        if a != b:
            break

    if a > b:
        n, p = a, b
    else:
        n, p = b, a

    return n, p


@pytest.fixture(
    scope="class",
    params=[
        _get_random_params(),
    ],
)
def spaces(request):
    n, p = request.param
    space = request.cls.space = Stiefel(n=n, p=p, equip=False)

    request.cls.testing_data.space = space


@pytest.mark.usefixtures("spaces")
class TestStiefel(StiefelTestCase, metaclass=GeometricDataBasedParametrizer):
    testing_data = StiefelTestData()


@pytest.mark.smoke
class TestStiefelStaticMethods(
    StiefelStaticMethodsTestCase, metaclass=DataBasedParametrizer
):
    Space = Stiefel
    testing_data = StiefelStaticMethodsTestData()


@pytest.fixture(
    scope="class",
    params=[
        _get_random_params(),
    ],
)
def equipped_spaces(request):
    n, p = request.param
    space = request.cls.space = Stiefel(n=n, p=p)

    request.cls.testing_data.space = space


@pytest.mark.usefixtures("equipped_spaces")
class TestStiefelCanonicalMetric(
    StiefelCanonicalMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = StiefelCanonicalMetricTestData(
        excluded_methods=(
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
        )
    )


class TestStiefelCanonicalMetricSquare(
    StiefelCanonicalMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    k = random.randint(2, 5)

    space = Stiefel(n=k, p=k, equip=False)
    space.equip_with_metric(StiefelCanonicalMetric)

    testing_data = StiefelCanonicalMetricSquareTestData(space)
