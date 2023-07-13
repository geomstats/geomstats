import random

import pytest

from geomstats.geometry.hermitian_matrices import HermitianMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import (
    ComplexVectorSpaceTestCase,
    MatrixVectorSpaceTestCaseMixins,
)

from .data.hermitian_matrices import HermitianMatricesTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = HermitianMatrices(n=request.param)


@pytest.mark.usefixtures("spaces")
class TestHermitianMatrices(
    MatrixVectorSpaceTestCaseMixins,
    ComplexVectorSpaceTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = HermitianMatricesTestData()
