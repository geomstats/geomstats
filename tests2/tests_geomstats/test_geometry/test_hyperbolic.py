import random

import pytest

from geomstats.geometry._hyperbolic import _Hyperbolic
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.hyperbolic import HyperbolicTestCase

from .data.hyperbolic import HyperbolicTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def dims(request):
    request.cls.dim = request.param
    request.cls.space = _Hyperbolic


@pytest.mark.usefixtures("dims")
class TestHyperbolic(HyperbolicTestCase, metaclass=DataBasedParametrizer):
    testing_data = HyperbolicTestData()
