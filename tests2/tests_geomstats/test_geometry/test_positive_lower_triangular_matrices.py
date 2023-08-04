import random

import pytest

from geomstats.geometry.positive_lower_triangular_matrices import (
    CholeskyMetric,
    PositiveLowerTriangularMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.positive_lower_triangular_matrices import (
    CholeskyMetricTestCase,
    PositiveLowerTriangularMatricesTestCase,
)
from tests2.tests_geomstats.test_geometry.data.positive_lower_triangular_matrices import (
    CholeskyMetricTestData,
)

from .data.positive_lower_triangular_matrices import (
    CholeskyMetric2TestData,
    PositiveLowerTriangularMatrices2TestData,
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
    request.cls.space = PositiveLowerTriangularMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestPositiveLowerTriangularMatrices(
    PositiveLowerTriangularMatricesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PositiveLowerTriangularMatricesTestData()


@pytest.mark.smoke
class TestPositiveLowerTriangularMatrices2(
    PositiveLowerTriangularMatricesTestCase, metaclass=DataBasedParametrizer
):
    space = PositiveLowerTriangularMatrices(n=2, equip=False)
    testing_data = PositiveLowerTriangularMatrices2TestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 3),
        random.randint(4, 5),
    ],
)
def equipped_spaces(request):
    space = request.cls.space = PositiveLowerTriangularMatrices(
        n=request.param, equip=False
    )
    space.equip_with_metric(CholeskyMetric)


@pytest.mark.usefixtures("equipped_spaces")
class TestCholeskyMetric(CholeskyMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = CholeskyMetricTestData()


@pytest.mark.smoke
class TestCholeskyMetric2(CholeskyMetricTestCase, metaclass=DataBasedParametrizer):
    space = PositiveLowerTriangularMatrices(n=2)
    testing_data = CholeskyMetric2TestData()
