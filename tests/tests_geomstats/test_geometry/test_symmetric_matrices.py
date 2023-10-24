import random

import pytest

from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import (
    MatrixVectorSpaceTestCaseMixins,
    VectorSpaceTestCase,
)
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase
from geomstats.test_cases.geometry.symmetric_matrices import (
    SymmetricMatricesOpsTestCase,
)

from .data.complex_matrices import ComplexMatricesMetricTestData
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
    request.cls.space = SymmetricMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestSymmetricMatrices(
    MatrixVectorSpaceTestCaseMixins,
    VectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = SymmetricMatricesTestData()


@pytest.mark.smoke
class TestSymmetricMatrices1(
    MatrixVectorSpaceTestCaseMixins,
    VectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    space = SymmetricMatrices(n=1, equip=False)
    testing_data = SymmetricMatrices1TestData()


@pytest.mark.smoke
class TestSymmetricMatrices2(VectorSpaceTestCase, metaclass=DataBasedParametrizer):
    space = SymmetricMatrices(n=2, equip=False)
    testing_data = SymmetricMatrices2TestData()


@pytest.mark.smoke
class TestSymmetricMatrices3(
    MatrixVectorSpaceTestCaseMixins,
    VectorSpaceTestCase,
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

    testing_data = ComplexMatricesMetricTestData()


@pytest.mark.smoke
class TestSymmetricMatricesOps(
    SymmetricMatricesOpsTestCase, metaclass=DataBasedParametrizer
):
    Space = SymmetricMatrices
    testing_data = SymmetricMatricesOpsTestData()
