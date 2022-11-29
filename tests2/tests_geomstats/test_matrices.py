import random

import pytest

from geomstats.geometry.matrices import Matrices
from geomstats.test.geometry.matrices import MatricesTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.matrices_data import MatricesTestData


@pytest.fixture(
    scope="class",
    params=[
        (2, 3),
        (random.randint(3, 5), random.randint(3, 5)),
    ],
)
def spaces(request):
    m, n = request.param
    request.cls.space = Matrices(m=m, n=n)


@pytest.mark.usefixtures("spaces")
class TestMatrices(MatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = MatricesTestData()
