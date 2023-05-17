import random

import pytest

from geomstats.geometry.hermitian import Hermitian
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.hermitian import HermitianTestCase

from .data.hermitian import HermitianTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = Hermitian(dim=request.param)


@pytest.mark.usefixtures("spaces")
class TestHermitian(HermitianTestCase, metaclass=DataBasedParametrizer):
    testing_data = HermitianTestData()
