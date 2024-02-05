import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.special_euclidean import (
    SpecialEuclidean,
    SpecialEuclideanMatricesLieAlgebra,
    _SpecialEuclideanMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.invariant_metric import InvariantMetricMatrixTestCase
from geomstats.test_cases.geometry.lie_algebra import MatrixLieAlgebraTestCase
from geomstats.test_cases.geometry.lie_group import MatrixLieGroupTestCase
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase
from geomstats.test_cases.geometry.special_euclidean import (
    SpecialEuclideanVectorsTestCase,
    homogeneous_representation_test_case,
    homogeneous_representation_vec_test_case,
)

from .data.lie_algebra import MatrixLieAlgebraTestData
from .data.matrices import MatricesMetricTestData
from .data.special_euclidean import (
    SpecialEuclidean2VectorsTestData,
    SpecialEuclideanMatrices2TestData,
    SpecialEuclideanMatricesCanonicalLeftMetricTestData,
    SpecialEuclideanMatricesLieAlgebra2TestData,
    SpecialEuclideanMatricesTestData,
    SpecialEuclideanVectorsTestData,
    homogeneous_representation_test_data,
)


@pytest.mark.parametrize("n_reps", random.sample(range(2, 5), 1))
@pytest.mark.parametrize("n", [2] + random.sample(range(3, 6), 1))
@pytest.mark.vec
def test_homogeneous_representation_vec(n, n_reps):
    return homogeneous_representation_vec_test_case(n, n_reps, atol=gs.atol)


@pytest.mark.parametrize(
    "rotation,translation,constant,expected", homogeneous_representation_test_data()
)
def test_homogeneous_representation(rotation, translation, constant, expected):
    return homogeneous_representation_test_case(
        rotation, translation, constant, expected, atol=gs.atol
    )


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces_mlg(request):
    request.cls.space = _SpecialEuclideanMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces_mlg")
class TestSpecialEuclideanMatrices(
    MatrixLieGroupTestCase, LevelSetTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SpecialEuclideanMatricesTestData()


@pytest.mark.smoke
class TestSpecialEuclideanMatrices2(
    MatrixLieGroupTestCase, LevelSetTestCase, metaclass=DataBasedParametrizer
):
    space = _SpecialEuclideanMatrices(n=2, equip=False)
    testing_data = SpecialEuclideanMatrices2TestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        3,
    ],
)
def spaces_vectors(request):
    request.cls.space = SpecialEuclidean(
        n=request.param, point_type="vector", equip=True
    )


@pytest.mark.usefixtures("spaces_vectors")
class TestSpecialEuclideanVectors(
    SpecialEuclideanVectorsTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SpecialEuclideanVectorsTestData()


@pytest.mark.smoke
class TestSpecialEuclidean2Vectors(
    SpecialEuclideanVectorsTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialEuclidean(n=2, point_type="vector", equip=False)
    testing_data = SpecialEuclidean2VectorsTestData()


class TestSpecialEuclideanMatricesCanonicalLeftMetric(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialEuclidean(random.randint(2, 3))
    testing_data = SpecialEuclideanMatricesCanonicalLeftMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces_mla(request):
    request.cls.space = SpecialEuclideanMatricesLieAlgebra(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces_mla")
class TestSpecialEuclideanMatricesLieAlgebra(
    MatrixLieAlgebraTestCase, metaclass=DataBasedParametrizer
):
    testing_data = MatrixLieAlgebraTestData()


@pytest.mark.parametrize("n,expected", [(2, 3), (3, 6), (10, 55)])
def test_dim_mla(n, expected):
    space = SpecialEuclideanMatricesLieAlgebra(n=n, equip=False)
    assert space.dim == expected


@pytest.mark.smoke
class TestSpecialEuclideanMatricesLieAlgebra2(
    MatrixLieAlgebraTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialEuclideanMatricesLieAlgebra(n=2, equip=False)
    testing_data = SpecialEuclideanMatricesLieAlgebra2TestData()


@pytest.mark.redundant
class TestMatricesMetric(MatricesMetricTestCase, metaclass=DataBasedParametrizer):
    n = random.randint(2, 5)
    space = SpecialEuclideanMatricesLieAlgebra(n=n, equip=True)
    testing_data = MatricesMetricTestData()
