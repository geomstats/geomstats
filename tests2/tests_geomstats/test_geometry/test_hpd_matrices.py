import random

import pytest

from geomstats.geometry.hpd_matrices import HPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.hpd_matrices import HPDMatricesTestCase

from .data.hpd_matrices import HPDMatricesTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = HPDMatrices(n=request.param)


@pytest.mark.usefixtures("spaces")
class TestHPDMatrices(HPDMatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = HPDMatricesTestData()
