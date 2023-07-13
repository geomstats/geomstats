import random

import pytest

from geomstats.geometry.siegel import Siegel
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import ComplexOpenSetTestCase

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
class TestSiegel(ComplexOpenSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = SiegelTestData()
