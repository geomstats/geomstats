import random

import pytest

from geomstats.geometry.special_euclidean import (
    SpecialEuclideanMatrixLieAlgebra,
    _SpecialEuclideanMatrices,
)
from geomstats.test.geometry.special_euclidean import (
    SpecialEuclideanMatricesTestCase,
    SpecialEuclideanMatrixLieAlgebraTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.special_euclidean_data import (
    SpecialEuclideanMatricesTestData,
    SpecialEuclideanMatrixLieAlgebra2TestData,
    SpecialEuclideanMatrixLieAlgebraTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        # random.randint(3, 5),
    ],
)
def spaces_mlg(request):
    request.cls.space = _SpecialEuclideanMatrices(n=request.param)


@pytest.mark.usefixtures("spaces_mlg")
class TestSpecialEuclideanMatrices(
    SpecialEuclideanMatricesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SpecialEuclideanMatricesTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces_mla(request):
    request.cls.space = SpecialEuclideanMatrixLieAlgebra(n=request.param)


@pytest.mark.usefixtures("spaces_mla")
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
