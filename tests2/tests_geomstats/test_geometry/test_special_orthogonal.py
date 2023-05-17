import random

import pytest

from geomstats.geometry.special_orthogonal import (
    SpecialOrthogonal,
    _SpecialOrthogonalMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.special_orthogonal import (
    SpecialOrthogonal2VectorsTestCase,
    SpecialOrthogonal3VectorsTestCase,
    SpecialOrthogonalMatricesTestCase,
)

from .data.special_orthogonal import (
    SpecialOrthogonal2VectorsTestData,
    SpecialOrthogonal3VectorsTestData,
    SpecialOrthogonalMatricesTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 3),
        random.randint(4, 6),
    ],
)
def mat_spaces(request):
    request.cls.space = _SpecialOrthogonalMatrices(n=request.param)


@pytest.mark.usefixtures("mat_spaces")
class TestSpecialOrthogonalMatrices(
    SpecialOrthogonalMatricesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SpecialOrthogonalMatricesTestData()


class TestSpecialOrthogonal2Vectors(
    SpecialOrthogonal2VectorsTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialOrthogonal(2, point_type="vector")
    testing_data = SpecialOrthogonal2VectorsTestData()


class TestSpecialOrthogonal3Vectors(
    SpecialOrthogonal3VectorsTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialOrthogonal(3, point_type="vector")
    testing_data = SpecialOrthogonal3VectorsTestData()
