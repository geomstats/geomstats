import random

import pytest

from geomstats.geometry.hermitian_matrices import HermitianMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import (
    ComplexVectorSpaceTestCase,
    MatrixVectorSpaceTestCaseMixins,
)
from geomstats.test_cases.geometry.hermitian import HermitianMetricTestCase
from geomstats.test_cases.geometry.symmetric_matrices import (
    SymmetricMatricesOpsTestCase,
)

from .data.complex_matrices import ComplexMatricesMetricTestData
from .data.hermitian_matrices import (
    HermitianMatrices2TestData,
    HermitianMatrices3TestData,
    HermitianMatricesOpsTestData,
    HermitianMatricesTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = HermitianMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestHermitianMatrices(
    MatrixVectorSpaceTestCaseMixins,
    ComplexVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = HermitianMatricesTestData()


@pytest.mark.smoke
class TestHermitianMatrices2(
    ComplexVectorSpaceTestCase, metaclass=DataBasedParametrizer
):
    space = HermitianMatrices(n=2, equip=False)
    testing_data = HermitianMatrices2TestData()


@pytest.mark.smoke
class TestHermitianMatrices3(
    MatrixVectorSpaceTestCaseMixins,
    ComplexVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    space = HermitianMatrices(n=3, equip=False)
    testing_data = HermitianMatrices3TestData()


@pytest.mark.redundant
class TestComplexMatricesMetric(
    HermitianMetricTestCase, metaclass=DataBasedParametrizer
):
    n = random.randint(2, 5)
    space = HermitianMatrices(n)
    testing_data = ComplexMatricesMetricTestData()


@pytest.mark.smoke
class TestHermitianMatricesOps(
    SymmetricMatricesOpsTestCase, metaclass=DataBasedParametrizer
):
    Space = HermitianMatrices
    testing_data = HermitianMatricesOpsTestData()
