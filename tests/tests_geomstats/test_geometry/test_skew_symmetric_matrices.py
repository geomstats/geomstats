import random

import pytest

from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
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


@pytest.fixture(
    scope="class",
    params=[
        2,
        3,
        random.randint(4, 9),
    ],
)
def spaces(request):
    request.cls.space = SkewSymmetricMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestSkewSymmetricMatrices(
    SkewSymmetricMatricesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SkewSymmetricMatricesTestData()


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
class TestMatricesMetric(MatricesMetricTestCase, metaclass=DataBasedParametrizer):
    n = random.randint(2, 5)
    space = SkewSymmetricMatrices(n=n)
    testing_data = MatricesMetricTestData()
