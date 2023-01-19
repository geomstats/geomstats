import random

import pytest

from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.test.geometry.poincare_half_space import PoincareHalfSpaceTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.poincare_half_space_data import PoincareHalfSpaceTestData


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
