import random

import pytest

from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.poincare_half_space import PoincareHalfSpaceTestCase

from .data.poincare_half_space import PoincareHalfSpaceTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = PoincareHalfSpace(dim=request.param)


@pytest.mark.usefixtures("spaces")
class TestPoincareHalfSpace(PoincareHalfSpaceTestCase, metaclass=DataBasedParametrizer):
    testing_data = PoincareHalfSpaceTestData()
