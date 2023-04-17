import random

import pytest

from geomstats.geometry.complex_matrices import ComplexMatrices
from geomstats.test.geometry.complex_matrices import ComplexMatricesTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.complex_matrices_data import ComplexMatricesTestData


@pytest.fixture(
    scope="class",
    params=[
        (2, 3),
        (random.randint(3, 5), random.randint(3, 5)),
    ],
)
def spaces(request):
    m, n = request.param
    request.cls.space = ComplexMatrices(m=m, n=n)


@pytest.mark.usefixtures("spaces")
class TestComplexMatrices(ComplexMatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = ComplexMatricesTestData()
