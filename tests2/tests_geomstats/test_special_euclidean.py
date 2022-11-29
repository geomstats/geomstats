import random

import pytest

from geomstats.geometry.special_euclidean import SpecialEuclideanMatrixLieAlgebra
from geomstats.test.geometry.special_euclidean import (
    SpecialEuclideanMatrixLieAlgebraTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.special_euclidean_data import (
    SpecialEuclideanMatrixLieAlgebra2TestData,
    SpecialEuclideanMatrixLieAlgebraTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = SpecialEuclideanMatrixLieAlgebra(n=request.param)


@pytest.mark.usefixtures("spaces")
class TestSpecialEuclideanMatrixLieAlgebra(
    SpecialEuclideanMatrixLieAlgebraTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SpecialEuclideanMatrixLieAlgebraTestData()


@pytest.mark.parametrize("n,expected", [(2, 3), (3, 6), (10, 55)])
def test_dim(n, expected):
    space = SpecialEuclideanMatrixLieAlgebra(n=n)
    assert space.dim == expected


class TestSpecialEuclideanMatrixLieAlgebra2(
    SpecialEuclideanMatrixLieAlgebraTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialEuclideanMatrixLieAlgebra(n=2)
    testing_data = SpecialEuclideanMatrixLieAlgebra2TestData()
