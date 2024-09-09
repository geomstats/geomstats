import random

import pytest

from geomstats.geometry.complex_matrices import ComplexMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.complex_matrices import ComplexMatricesTestCase
from geomstats.test_cases.geometry.hermitian import HermitianMetricTestCase

from .data.base import ComplexMatrixVectorSpaceTestData
from .data.complex_matrices import (
    ComplexMatrices33TestData,
    ComplexMatricesMetricTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        (2, 3),
        (random.randint(3, 5), random.randint(3, 5)),
    ],
)
def spaces(request):
    m, n = request.param
    request.cls.space = ComplexMatrices(m=m, n=n, equip=False)


@pytest.mark.usefixtures("spaces")
class TestComplexMatrices(ComplexMatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = ComplexMatrixVectorSpaceTestData()


@pytest.mark.smoke
class TestComplexMatrices33(ComplexMatricesTestCase, metaclass=DataBasedParametrizer):
    space = ComplexMatrices(m=3, n=3, equip=False)
    testing_data = ComplexMatrices33TestData()


@pytest.fixture(
    scope="class",
    params=[
        (2, 3),
        (random.randint(3, 5), random.randint(3, 5)),
    ],
)
def equipped_spaces(request):
    m, n = request.param
    request.cls.space = ComplexMatrices(m=m, n=n)


@pytest.mark.usefixtures("equipped_spaces")
class TestComplexMatricesMetric(
    HermitianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ComplexMatricesMetricTestData()
