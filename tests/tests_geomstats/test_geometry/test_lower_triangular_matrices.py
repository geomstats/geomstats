import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.lower_triangular_matrices import (
    LowerTriangularMatrices,
    StrictlyLowerTriangularMatrices,
)
from geomstats.test_cases.geometry.base import (
    LevelSetTestCase,
    MatrixVectorSpaceTestCase,
)
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase

from .data.base import MatrixVectorSpaceTestData
from .data.lower_triangular_matrices import (
    LowerTriangularMatrices2TestData,
    LowerTriangularMatrices3TestData,
    StrictlyLowerTriangularMatricesTestData,
)
from .data.matrices import MatricesMetricTestData


class TestLowerTriangularMatrices(
    MatrixVectorSpaceTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n = random.randint(2, 5)
    space = LowerTriangularMatrices(n=_n, equip=False)

    testing_data = MatrixVectorSpaceTestData(space)


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
class TestMatricesMetric(
    MatricesMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = LowerTriangularMatrices(n=random.randint(3, 5))
    testing_data = MatricesMetricTestData(space)


class TestStrictlyLowerTriangularMatrices(
    LevelSetTestCase,
    MatrixVectorSpaceTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n = random.randint(2, 6)
    space = StrictlyLowerTriangularMatrices(n=_n, equip=False)

    testing_data = StrictlyLowerTriangularMatricesTestData(space)
