import random

import pytest

from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import MatrixVectorSpaceTestCase
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase

from .data.lower_triangular_matrices import (
    LowerTriangularMatrices2TestData,
    LowerTriangularMatrices3TestData,
    LowerTriangularMatricesTestData,
)
from .data.matrices import MatricesMetricTestData


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
    MatrixVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = LowerTriangularMatricesTestData()


@pytest.mark.smoke
class TestLowerTriangularMatrices2(
    MatrixVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    space = LowerTriangularMatrices(n=2)
    testing_data = LowerTriangularMatrices2TestData()


@pytest.mark.smoke
class TestLowerTriangularMatrices3(
    MatrixVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    space = LowerTriangularMatrices(n=3)
    testing_data = LowerTriangularMatrices3TestData()


@pytest.mark.redundant
class TestMatricesMetric(MatricesMetricTestCase, metaclass=DataBasedParametrizer):
    space = LowerTriangularMatrices(n=random.randint(3, 5))
    testing_data = MatricesMetricTestData()
