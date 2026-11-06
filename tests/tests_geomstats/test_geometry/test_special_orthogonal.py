import random

import pytest

from geomstats.geometry.special_orthogonal import (
    SpecialOrthogonal,
    _SpecialOrthogonalMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.special_orthogonal import (
    SpecialOrthogonal3VectorsTestCase,
    SpecialOrthogonalMatricesTestCase,
    SpecialOrthogonalVectorsTestCase,
)

from .data.special_orthogonal import (
    SpecialOrthogonal2VectorsSmokeTestData,
    SpecialOrthogonal2VectorsTestData,
    SpecialOrthogonal3VectorsSmokeTestData,
    SpecialOrthogonal3VectorsTestData,
    SpecialOrthogonalMatrices2TestData,
    SpecialOrthogonalMatrices3TestData,
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
    request.cls.space = _SpecialOrthogonalMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("mat_spaces")
class TestSpecialOrthogonalMatrices(
    SpecialOrthogonalMatricesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = SpecialOrthogonalMatricesTestData()


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


class TestSpecialOrthogonal2Vectors(
    SpecialOrthogonalVectorsTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialOrthogonal(2, point_type="vector", equip=False)
    testing_data = SpecialOrthogonal2VectorsTestData()


@pytest.mark.smoke
class TestSpecialOrthogonal2VectorsSmoke(
    SpecialOrthogonalVectorsTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialOrthogonal(2, point_type="vector", equip=False)
    testing_data = SpecialOrthogonal2VectorsSmokeTestData()


class TestSpecialOrthogonal3Vectors(
    SpecialOrthogonal3VectorsTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialOrthogonal(3, point_type="vector", equip=True)
    testing_data = SpecialOrthogonal3VectorsTestData()


@pytest.mark.smoke
class TestSpecialOrthogonal3VectorsSmoke(
    SpecialOrthogonal3VectorsTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialOrthogonal(3, point_type="vector", equip=True)
    testing_data = SpecialOrthogonal3VectorsSmokeTestData()


def test_so2_vectors_exp_log_default_to_identity():
    import geomstats.backend as gs

    space = SpecialOrthogonal(2, point_type="vector")
    tangent_vec = gs.array([[0.3], [-1.2]])
    point = space.exp(tangent_vec)
    assert gs.allclose(point, space.exp(tangent_vec, space.identity))
    assert gs.allclose(space.log(point), space.log(point, space.identity))
    assert gs.allclose(space.log(point), tangent_vec)
