import random

import pytest

from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.skew_symmetric_matrices import (
    SkewSymmetricMatricesTestCase,
)

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
    request.cls.space = SkewSymmetricMatrices(n=request.param)


@pytest.mark.usefixtures("spaces")
class TestSkewSymmetricMatrices(
    SkewSymmetricMatricesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SkewSymmetricMatricesTestData()


@pytest.mark.smoke
class TestSkewSymmetricMatrices2(
    SkewSymmetricMatricesTestCase, metaclass=DataBasedParametrizer
):
    space = SkewSymmetricMatrices(n=2)
    testing_data = SkewSymmetricMatrices2TestData()


@pytest.mark.smoke
class TestSkewSymmetricMatrices3(
    SkewSymmetricMatricesTestCase, metaclass=DataBasedParametrizer
):
    space = SkewSymmetricMatrices(n=3)
    testing_data = SkewSymmetricMatrices3TestData()
