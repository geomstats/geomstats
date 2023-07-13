import random

import pytest

from geomstats.geometry.hpd_matrices import HPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import ComplexOpenSetTestCase
from geomstats.test_cases.geometry.spd_matrices import SPDMatricesTestCaseMixins

from .data.hpd_matrices import HPDMatricesTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = HPDMatrices(n=request.param)


@pytest.mark.usefixtures("spaces")
class TestHPDMatrices(
    SPDMatricesTestCaseMixins, ComplexOpenSetTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HPDMatricesTestData()
