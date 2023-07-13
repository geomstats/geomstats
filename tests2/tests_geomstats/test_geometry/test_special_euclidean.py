import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.special_euclidean import (
    SpecialEuclidean,
    SpecialEuclideanMatrixCanonicalLeftMetric,
    SpecialEuclideanMatrixLieAlgebra,
    _SpecialEuclideanMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.invariant_metric import InvariantMetricMatrixTestCase
from geomstats.test_cases.geometry.lie_algebra import MatrixLieAlgebraTestCase
from geomstats.test_cases.geometry.lie_group import MatrixLieGroupTestCase
from geomstats.test_cases.geometry.special_euclidean import (
    SpecialEuclideanVectorsTestCase,
    homogeneous_representation_test_case,
    homogeneous_representation_vec_test_case,
)

from .data.special_euclidean import (
    SpecialEuclideanMatricesTestData,
    SpecialEuclideanMatrixCanonicalLeftMetricTestData,
    SpecialEuclideanMatrixLieAlgebra2TestData,
    SpecialEuclideanMatrixLieAlgebraTestData,
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
        random.randint(3, 6),
    ],
)
def spaces_mlg(request):
    request.cls.space = _SpecialEuclideanMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces_mlg")
class TestSpecialEuclideanMatrices(
    MatrixLieGroupTestCase, LevelSetTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SpecialEuclideanMatricesTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        3,
    ],
)
def spaces_vectors(request):
    request.cls.space = SpecialEuclidean(n=request.param, point_type="vector")


@pytest.mark.usefixtures("spaces_vectors")
class TestSpecialEuclideanVectors(
    SpecialEuclideanVectorsTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SpecialEuclideanVectorsTestData()


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
    MatrixLieAlgebraTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SpecialEuclideanMatrixLieAlgebraTestData()


@pytest.mark.parametrize("n,expected", [(2, 3), (3, 6), (10, 55)])
def test_dim_mla(n, expected):
    space = SpecialEuclideanMatrixLieAlgebra(n=n)
    assert space.dim == expected


@pytest.mark.smoke
class TestSpecialEuclideanMatrixLieAlgebra2(
    MatrixLieAlgebraTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialEuclideanMatrixLieAlgebra(n=2)
    testing_data = SpecialEuclideanMatrixLieAlgebra2TestData()


@pytest.fixture(
    scope="class",
    params=[
        SpecialEuclidean(random.randint(2, 3), equip=False),
    ],
)
def equipped_SE_matrix_groups(request):
    space = request.cls.space = request.param
    space.equip_with_metric(SpecialEuclideanMatrixCanonicalLeftMetric)


@pytest.mark.usefixtures("equipped_SE_matrix_groups")
class TestSpecialEuclideanMatrixCanonicalLeftMetric(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SpecialEuclideanMatrixCanonicalLeftMetricTestData()
