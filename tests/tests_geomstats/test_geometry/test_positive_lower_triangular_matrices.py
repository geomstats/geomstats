import random

import pytest

from geomstats.geometry.positive_lower_triangular_matrices import (
    CholeskyMetric,
    PositiveLowerTriangularMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.positive_lower_triangular_matrices import (
    CholeskyMetricTestCase,
)

from .data.base import VectorSpaceOpenSetTestData
from .data.positive_lower_triangular_matrices import (
    CholeskyMetric2TestData,
    CholeskyMetricTestData,
    PositiveLowerTriangularMatrices2TestData,
)


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
    ],
)
def spaces(request):
    request.cls.space = PositiveLowerTriangularMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestPositiveLowerTriangularMatrices(
    VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer
):
    testing_data = VectorSpaceOpenSetTestData()


@pytest.mark.smoke
class TestPositiveLowerTriangularMatrices2(
    VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = PositiveLowerTriangularMatrices(n=2, equip=False)
    testing_data = PositiveLowerTriangularMatrices2TestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
    ],
)
def equipped_spaces(request):
    request.cls.space = PositiveLowerTriangularMatrices(n=request.param)


@pytest.mark.usefixtures("equipped_spaces")
class TestCholeskyMetric(CholeskyMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = CholeskyMetricTestData()


@pytest.mark.smoke
class TestCholeskyMetric2(CholeskyMetricTestCase, metaclass=DataBasedParametrizer):
    space = PositiveLowerTriangularMatrices(n=2, equip=False)
    space.equip_with_metric(CholeskyMetric)
    testing_data = CholeskyMetric2TestData()
