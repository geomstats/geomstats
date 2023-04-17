import random

import pytest

from geomstats.geometry.general_linear import SquareMatrices
from geomstats.test.geometry.square_matrices import SquareMatricesTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.square_matrices_data import (
    SquareMatrices3TestData,
    SquareMatricesTestData,
)


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
class TestSquareMatrices(SquareMatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = SquareMatricesTestData()


class TestSquareMatrices3(SquareMatricesTestCase, metaclass=DataBasedParametrizer):
    space = SquareMatrices(n=3)
    testing_data = SquareMatrices3TestData()
