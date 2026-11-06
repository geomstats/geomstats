import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase
from geomstats.test_cases.geometry.skew_symmetric_matrices import (
    SkewSymmetricMatricesTestCase,
)

from .data.matrices import MatricesMetricTestData
from .data.skew_symmetric_matrices import (
    SkewSymmetricMatrices2TestData,
    SkewSymmetricMatrices3TestData,
    SkewSymmetricMatricesTestData,
)


class TestSkewSymmetricMatrices(
    SkewSymmetricMatricesTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = SkewSymmetricMatrices(n=random.randint(3, 9), equip=False)
    testing_data = SkewSymmetricMatricesTestData(space)


@pytest.mark.smoke
class TestSkewSymmetricMatrices2(
    SkewSymmetricMatricesTestCase, metaclass=DataBasedParametrizer
):
    space = SkewSymmetricMatrices(n=2, equip=False)
    testing_data = SkewSymmetricMatrices2TestData()


@pytest.mark.smoke
class TestSkewSymmetricMatrices3(
    SkewSymmetricMatricesTestCase, metaclass=DataBasedParametrizer
):
    space = SkewSymmetricMatrices(n=3, equip=False)
    testing_data = SkewSymmetricMatrices3TestData()


@pytest.mark.redundant
class TestMatricesMetric(
    MatricesMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = SkewSymmetricMatrices(n=random.randint(2, 5))
    testing_data = MatricesMetricTestData(space)
