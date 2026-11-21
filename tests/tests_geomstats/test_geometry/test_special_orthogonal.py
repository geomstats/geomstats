import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.special_orthogonal import (
    SpecialOrthogonal,
    _SpecialOrthogonalMatrices,
)
from geomstats.test_cases.geometry.special_orthogonal import (
    SpecialOrthogonal3VectorsTestCase,
    SpecialOrthogonalMatricesTestCase,
    SpecialOrthogonalVectorsTestCase,
)

from .data.special_orthogonal import (
    SpecialOrthogonal2VectorsTestData,
    SpecialOrthogonal3VectorsTestData,
    SpecialOrthogonalMatrices2TestData,
    SpecialOrthogonalMatrices3TestData,
    SpecialOrthogonalMatricesTestData,
)


@pytest.mark.smoke
class TestSpecialOrthogonalMatrices2(
    SpecialOrthogonalMatricesTestCase, metaclass=DataBasedParametrizer
):
    space = _SpecialOrthogonalMatrices(n=2, equip=False)
    testing_data = SpecialOrthogonalMatrices2TestData()


@pytest.mark.smoke
class TestSpecialOrthogonalMatrices3(
    SpecialOrthogonalMatricesTestCase, metaclass=DataBasedParametrizer
):
    space = _SpecialOrthogonalMatrices(n=3, equip=False)
    testing_data = SpecialOrthogonalMatrices3TestData()


class TestSpecialOrthogonalMatrices(
    SpecialOrthogonalMatricesTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = _SpecialOrthogonalMatrices(n=random.randint(2, 3), equip=False)
    testing_data = SpecialOrthogonalMatricesTestData(space)


class TestSpecialOrthogonalHighDimMatrices(
    SpecialOrthogonalMatricesTestCase, metaclass=GeometricDataBasedParametrizer
):
    """Special orthogonal matrices without rotation vectors, for n > 3."""

    space = _SpecialOrthogonalMatrices(n=random.randint(4, 6), equip=False)
    testing_data = SpecialOrthogonalMatricesTestData(
        space,
        excluded_methods=(
            "rotation_vector_from_matrix",
            "matrix_from_rotation_vector",
            "rotation_vector_from_matrix_after_matrix_from_rotation_vector",
            "matrix_from_rotation_vector_after_rotation_vector_from_matrix",
        ),
    )


class TestSpecialOrthogonal2Vectors(
    SpecialOrthogonalVectorsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = SpecialOrthogonal(2, point_type="vector", equip=False)
    testing_data = SpecialOrthogonal2VectorsTestData(space)


class TestSpecialOrthogonal3Vectors(
    SpecialOrthogonal3VectorsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = SpecialOrthogonal(3, point_type="vector", equip=True)
    testing_data = SpecialOrthogonal3VectorsTestData(space)
