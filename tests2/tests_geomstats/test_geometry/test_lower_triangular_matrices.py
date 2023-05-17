import random

import pytest

from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.lower_triangular_matrices import (
    LowerTriangularMatricesTestCase,
)

from .data.lower_triangular_matrices import (
    LowerTriangularMatrices2TestData,
    LowerTriangularMatrices3TestData,
    LowerTriangularMatricesTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = LowerTriangularMatrices(n=request.param)


@pytest.mark.usefixtures("spaces")
class TestLowerTriangularMatrices(
    LowerTriangularMatricesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = LowerTriangularMatricesTestData()


class TestLowerTriangularMatrices2(
    LowerTriangularMatricesTestCase, metaclass=DataBasedParametrizer
):
    space = LowerTriangularMatrices(n=2)
    testing_data = LowerTriangularMatrices2TestData()


class TestLowerTriangularMatrices3(
    LowerTriangularMatricesTestCase, metaclass=DataBasedParametrizer
):
    space = LowerTriangularMatrices(n=3)
    testing_data = LowerTriangularMatrices3TestData()
