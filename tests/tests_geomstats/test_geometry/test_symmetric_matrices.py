import random

import pytest

from geomstats.geometry.symmetric_matrices import (
    SymmetricHollowMatrices,
    SymmetricMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import (
    LevelSetTestCase,
    MatrixVectorSpaceTestCase,
)
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase

from .data.matrices import MatricesMetricTestData
from .data.symmetric_matrices import (
    SymmetricHollowMatricesTestData,
    SymmetricMatrices1TestData,
    SymmetricMatrices2TestData,
    SymmetricMatrices3TestData,
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
    request.cls.space = SymmetricMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestSymmetricMatrices(
    MatrixVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = SymmetricMatricesTestData()


@pytest.mark.smoke
class TestSymmetricMatrices1(
    MatrixVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    space = SymmetricMatrices(n=1, equip=False)
    testing_data = SymmetricMatrices1TestData()


@pytest.mark.smoke
class TestSymmetricMatrices2(
    MatrixVectorSpaceTestCase, metaclass=DataBasedParametrizer
):
    space = SymmetricMatrices(n=2, equip=False)
    testing_data = SymmetricMatrices2TestData()


@pytest.mark.smoke
class TestSymmetricMatrices3(
    MatrixVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    space = SymmetricMatrices(n=3, equip=False)
    testing_data = SymmetricMatrices3TestData()


@pytest.mark.parametrize("n,expected", [(1, 1), (2, 3), (5, 15)])
def test_dim(n, expected):
    space = SymmetricMatrices(n, equip=False)
    assert space.dim == expected


@pytest.mark.redundant
class TestMatricesMetric(MatricesMetricTestCase, metaclass=DataBasedParametrizer):
    n = random.randint(3, 5)
    space = SymmetricMatrices(n=n)

    testing_data = MatricesMetricTestData()


class TestSymmetricHollowMatrices(
    MatrixVectorSpaceTestCase,
    LevelSetTestCase,
    metaclass=DataBasedParametrizer,
):
    _n = random.randint(2, 5)
    space = SymmetricHollowMatrices(n=_n, equip=False)

    testing_data = SymmetricHollowMatricesTestData()
