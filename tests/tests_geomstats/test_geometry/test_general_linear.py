import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.general_linear import (
    GeneralLinear,
    PositiveGeneralLinear,
    SquareMatrices,
)
from geomstats.test_cases.geometry.general_linear import GeneralLinearTestCase
from geomstats.test_cases.geometry.lie_algebra import MatrixLieAlgebraTestCase
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase

from .data.general_linear import (
    GeneralLinear2TestData,
    GeneralLinear3TestData,
    GeneralLinearMatricesMetricTestData,
    GeneralLinearTestData,
    SquareMatrices3TestData,
)
from .data.lie_algebra import MatrixLieAlgebraTestData
from .data.matrices import MatricesMetricTestData


class TestGeneralLinear(
    GeneralLinearTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = GeneralLinear(n=random.randint(2, 5), equip=False)
    testing_data = GeneralLinearTestData(space)


class TestPositiveGeneralLinear(
    GeneralLinearTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveGeneralLinear(n=random.randint(2, 5), equip=False)
    testing_data = GeneralLinearTestData(space)


@pytest.mark.smoke
class TestGeneralLinear2(GeneralLinearTestCase, metaclass=DataBasedParametrizer):
    space = GeneralLinear(n=2, equip=False)
    testing_data = GeneralLinear2TestData()


@pytest.mark.smoke
class TestGeneralLinear3(GeneralLinearTestCase, metaclass=DataBasedParametrizer):
    space = GeneralLinear(n=3, equip=False)
    testing_data = GeneralLinear3TestData()


@pytest.mark.redundant
class TestGeneralLinearMatricesMetric(
    MatricesMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = GeneralLinear(n=random.randint(2, 5))
    testing_data = GeneralLinearMatricesMetricTestData(space)


@pytest.mark.redundant
class TestPositiveGeneralLinearMatricesMetric(
    MatricesMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveGeneralLinear(n=random.randint(2, 5))
    testing_data = GeneralLinearMatricesMetricTestData(space)


class TestSquareMatrices(
    MatrixLieAlgebraTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = SquareMatrices(n=random.randint(2, 5), equip=False)
    testing_data = MatrixLieAlgebraTestData(space)


@pytest.mark.smoke
class TestSquareMatrices3(MatrixLieAlgebraTestCase, metaclass=DataBasedParametrizer):
    space = SquareMatrices(n=3, equip=False)
    testing_data = SquareMatrices3TestData()


@pytest.mark.redundant
class TestSquareMatricesMetric(
    MatricesMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = SquareMatrices(n=random.randint(2, 5))
    testing_data = MatricesMetricTestData(space)
