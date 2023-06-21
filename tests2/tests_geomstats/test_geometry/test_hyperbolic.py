import random

import pytest

from geomstats.geometry._hyperbolic import _Hyperbolic
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.hyperbolic import HyperbolicCoordsTransformTestCase

from .data.hyperbolic import (
    HyperbolicCoordsTransform2TestData,
    HyperbolicCoordsTransformTestData,
)


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
class TestHyperbolicCoordsTransform(
    HyperbolicCoordsTransformTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HyperbolicCoordsTransformTestData()


@pytest.mark.random
class TestHyperbolicCoordsTransform2(
    HyperbolicCoordsTransformTestCase, metaclass=DataBasedParametrizer
):
    space = _Hyperbolic
    testing_data = HyperbolicCoordsTransform2TestData()
