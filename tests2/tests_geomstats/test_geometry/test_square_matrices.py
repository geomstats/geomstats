import random

import pytest

from geomstats.geometry.general_linear import SquareMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.lie_algebra import MatrixLieAlgebraTestCase

from .data.square_matrices import SquareMatrices3TestData, SquareMatricesTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = SquareMatrices(n=request.param)


@pytest.mark.usefixtures("spaces")
class TestSquareMatrices(MatrixLieAlgebraTestCase, metaclass=DataBasedParametrizer):
    testing_data = SquareMatricesTestData()


class TestSquareMatrices3(MatrixLieAlgebraTestCase, metaclass=DataBasedParametrizer):
    space = SquareMatrices(n=3)
    testing_data = SquareMatrices3TestData()
