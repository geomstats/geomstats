import random

import pytest

from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.geometry.spd_matrices import SPDMatricesTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.spd_matrices_data import SPDMatricesTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = SPDMatrices(n=request.param)


@pytest.mark.usefixtures("spaces")
class TestSPDMatrices(SPDMatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = SPDMatricesTestData()
