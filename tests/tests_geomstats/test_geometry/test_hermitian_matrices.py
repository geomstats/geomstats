import random

import pytest

from geomstats.geometry.hermitian_matrices import HermitianMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import ComplexMatrixVectorSpaceTestCase
from geomstats.test_cases.geometry.hermitian import HermitianMetricTestCase

from .data.complex_matrices import ComplexMatricesMetricTestData
from .data.hermitian_matrices import (
    HermitianMatrices2TestData,
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
    ComplexMatrixVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = HermitianMatricesTestData()


@pytest.mark.smoke
class TestHermitianMatrices2(
    ComplexMatrixVectorSpaceTestCase, metaclass=DataBasedParametrizer
):
    space = HermitianMatrices(n=2, equip=False)
    testing_data = HermitianMatrices2TestData()


@pytest.mark.redundant
class TestComplexMatricesMetric(
    HermitianMetricTestCase, metaclass=DataBasedParametrizer
):
    n = random.randint(2, 5)
    space = HermitianMatrices(n)
    testing_data = ComplexMatricesMetricTestData()
