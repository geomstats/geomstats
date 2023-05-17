import random

import pytest

from geomstats.geometry.siegel import Siegel
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.siegel import SiegelTestCase

from .data.siegel import SiegelTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = Siegel(request.param)


@pytest.mark.usefixtures("spaces")
class TestSiegel(SiegelTestCase, metaclass=DataBasedParametrizer):
    testing_data = SiegelTestData()
