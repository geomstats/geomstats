import random

import pytest

from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.test.geometry.poincare_ball import PoincareBallTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.poincare_ball_data import PoincareBallTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = PoincareBall(dim=request.param)


@pytest.mark.usefixtures("spaces")
class TestPoincareBall(PoincareBallTestCase, metaclass=DataBasedParametrizer):
    testing_data = PoincareBallTestData()
