import random

import pytest

from geomstats.geometry.hermitian_matrices import HermitianMatrices
from geomstats.test.geometry.hermitian_matrices import HermitianMatricesTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.hermitian_matrices_data import HermitianMatricesTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = HermitianMatrices(n=request.param)


@pytest.mark.usefixtures("spaces")
class TestHermitianMatrices(HermitianMatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = HermitianMatricesTestData()
