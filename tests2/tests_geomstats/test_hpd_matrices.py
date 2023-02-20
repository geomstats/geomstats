import random

import pytest

from geomstats.geometry.hpd_matrices import HPDMatrices
from geomstats.test.geometry.hpd_matrices import HPDMatricesTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.hpd_matrices_data import HPDMatricesTestData


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
