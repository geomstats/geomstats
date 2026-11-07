import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

import geomstats.backend as gs
from geomstats.geometry.special_euclidean import (
    SpecialEuclidean,
    SpecialEuclideanMatricesLieAlgebra,
    _SpecialEuclideanMatrices,
)
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.invariant_metric import InvariantMetricMatrixTestCase
from geomstats.test_cases.geometry.lie_algebra import MatrixLieAlgebraTestCase
from geomstats.test_cases.geometry.lie_group import MatrixLieGroupTestCase
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase
from geomstats.test_cases.geometry.special_euclidean import (
    SpecialEuclideanVectorsTestCase,
    homogeneous_representation_test_case,
)

from .data.invariant_metric import InvariantMetricMatrixTestData
from .data.lie_algebra import MatrixLieAlgebraTestData
from .data.matrices import MatricesMetricTestData
from .data.special_euclidean import (
    SpecialEuclidean2VectorsTestData,
    SpecialEuclideanMatrices2TestData,
    SpecialEuclideanMatricesLieAlgebra2TestData,
    SpecialEuclideanMatricesTestData,
    SpecialEuclideanVectorsTestData,
    homogeneous_representation_test_data,
)


@pytest.mark.parametrize(
    "rotation,translation,constant,expected", homogeneous_representation_test_data()
)
def test_homogeneous_representation(rotation, translation, constant, expected):
    return homogeneous_representation_test_case(
        rotation, translation, constant, expected, atol=gs.atol
    )


@pytest.mark.smoke
class TestSpecialEuclideanMatrices2(
    MatrixLieGroupTestCase, LevelSetTestCase, metaclass=DataBasedParametrizer
):
    space = _SpecialEuclideanMatrices(n=2, equip=False)
    testing_data = SpecialEuclideanMatrices2TestData()


class TestSpecialEuclideanMatrices(
    MatrixLieGroupTestCase, LevelSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = _SpecialEuclideanMatrices(n=random.randint(2, 5), equip=False)
    testing_data = SpecialEuclideanMatricesTestData(space)


class TestSpecialEuclidean2Vectors(
    SpecialEuclideanVectorsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = SpecialEuclidean(n=2, point_type="vector", equip=True)
    testing_data = SpecialEuclidean2VectorsTestData(
        space,
        excluded_methods=(
            "lie_bracket",
            "lie_bracket_belongs",
            "lie_bracket_antisymmetry",
            "lie_bracket_jacobi_identity",
            "lie_bracket_bilinearity",
        ),
    )


class TestSpecialEuclidean3Vectors(
    SpecialEuclideanVectorsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = SpecialEuclidean(n=3, point_type="vector", equip=True)
    testing_data = SpecialEuclideanVectorsTestData(
        space,
        excluded_methods=(
            "lie_bracket",
            "lie_bracket_belongs",
            "lie_bracket_antisymmetry",
            "lie_bracket_jacobi_identity",
            "lie_bracket_bilinearity",
        ),
    )


class TestSpecialEuclideanMatricesCanonicalLeftMetric(
    InvariantMetricMatrixTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = SpecialEuclidean(random.randint(2, 3))
    testing_data = InvariantMetricMatrixTestData(
        space, excluded_methods=("injectivity_radius",)
    )


@pytest.mark.smoke
class TestSpecialEuclideanMatricesLieAlgebra2(
    MatrixLieAlgebraTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialEuclideanMatricesLieAlgebra(n=2, equip=False)
    testing_data = SpecialEuclideanMatricesLieAlgebra2TestData()


class TestSpecialEuclideanMatricesLieAlgebra(
    MatrixLieAlgebraTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = SpecialEuclideanMatricesLieAlgebra(n=random.randint(2, 5), equip=False)
    testing_data = MatrixLieAlgebraTestData(space)


@pytest.mark.redundant
class TestMatricesMetric(
    MatricesMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = SpecialEuclideanMatricesLieAlgebra(n=random.randint(2, 5), equip=True)
    testing_data = MatricesMetricTestData(space)
