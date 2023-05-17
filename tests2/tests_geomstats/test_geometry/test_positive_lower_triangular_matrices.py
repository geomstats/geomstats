import random

import pytest

from geomstats.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatricesTestCase,
)

from .data.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatricesTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 3),
        random.randint(4, 5),
    ],
)
def spaces(request):
    request.cls.space = PositiveLowerTriangularMatrices(n=request.param)


@pytest.mark.usefixtures("spaces")
class TestPositiveLowerTriangularMatrices(
    PositiveLowerTriangularMatricesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PositiveLowerTriangularMatricesTestData()
