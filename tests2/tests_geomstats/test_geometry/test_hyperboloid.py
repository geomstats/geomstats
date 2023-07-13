import random

import pytest

from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import LevelSetTestCase

from .data.hyperboloid import HyperboloidTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = Hyperboloid(dim=request.param)


@pytest.mark.usefixtures("spaces")
class TestHyperboloid(LevelSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = HyperboloidTestData()
