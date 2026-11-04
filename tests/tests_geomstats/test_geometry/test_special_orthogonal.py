import random

import pytest

import geomstats.backend as gs
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


@pytest.mark.parametrize("angle", [1e-3, 1e-5, 3e-6, 1e-6, 1e-7])
def test_log_of_close_rotations(angle):
    # the logarithm of two close rotations must not be mistaken for antipodal
    space = SpecialOrthogonal(3, point_type="matrix")
    rotation_from_vec = SpecialOrthogonal(
        3, point_type="vector", equip=False
    ).matrix_from_rotation_vector

    base_point = rotation_from_vec(gs.array([-1.5, -1.0, -1.0]))
    point = gs.matmul(base_point, rotation_from_vec(gs.array([angle, 0.0, 0.0])))

    tangent_vec = space.metric.log(point, base_point)
    assert gs.allclose(space.metric.exp(tangent_vec, base_point), point, atol=1e-10)
    assert gs.allclose(space.metric.dist(point, base_point), angle, rtol=1e-3)
