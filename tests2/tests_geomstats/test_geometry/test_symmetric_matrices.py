import random

import pytest

from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.symmetric_matrices import (
    SymmetricMatricesOpsTestCase,
    SymmetricMatricesTestCase,
)

from .data.symmetric_matrices import (
    SymmetricMatrices1TestData,
    SymmetricMatrices2TestData,
    SymmetricMatrices3TestData,
    SymmetricMatricesOpsTestData,
    SymmetricMatricesTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = SymmetricMatrices(n=request.param)


@pytest.mark.usefixtures("spaces")
class TestSymmetricMatrices(SymmetricMatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = SymmetricMatricesTestData()


class TestSymmetricMatrices1(
    SymmetricMatricesTestCase, metaclass=DataBasedParametrizer
):
    space = SymmetricMatrices(n=1)
    testing_data = SymmetricMatrices1TestData()


class TestSymmetricMatrices2(
    SymmetricMatricesTestCase, metaclass=DataBasedParametrizer
):
    space = SymmetricMatrices(n=2)
    testing_data = SymmetricMatrices2TestData()


class TestSymmetricMatrices3(
    SymmetricMatricesTestCase, metaclass=DataBasedParametrizer
):
    space = SymmetricMatrices(n=3)
    testing_data = SymmetricMatrices3TestData()


@pytest.mark.parametrize("n,expected", [(1, 1), (2, 3), (5, 15)])
def test_dim(n, expected):
    space = SymmetricMatrices(n=n)
    assert space.dim == expected


class TestSymmetricMatricesOps(
    SymmetricMatricesOpsTestCase, metaclass=DataBasedParametrizer
):
    Space = SymmetricMatrices
    testing_data = SymmetricMatricesOpsTestData()
