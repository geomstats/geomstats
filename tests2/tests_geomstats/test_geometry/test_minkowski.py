import random

import pytest

from geomstats.geometry.minkowski import Minkowski
from geomstats.test.geometry.minkowski import MinkowskiTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.minkowski_data import MinkowskiTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = Minkowski(dim=request.param)


@pytest.mark.usefixtures("spaces")
class TestMinkowski(MinkowskiTestCase, metaclass=DataBasedParametrizer):
    # bare minimum testing as Euclidean covers all cases
    testing_data = MinkowskiTestData()
