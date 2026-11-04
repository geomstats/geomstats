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


@pytest.mark.smoke
def test_dist_between_nearby_rotation_matrices():
    """Distance between rotations 1e-6 rad apart is defined and equal to 1e-6.

    Nearby rotations used to be flagged as antipodal, which made the log raise.
    See https://github.com/geomstats/geomstats/issues/2168.
    """
    space = SpecialOrthogonal(3, point_type="matrix")
    rot = SpecialOrthogonal(
        3, point_type="vector", equip=False
    ).matrix_from_rotation_vector
    base_point = rot(gs.array([-2.0, -2.0, -0.5]))
    point = gs.matmul(base_point, rot(gs.array([0.0, 0.0, 1e-6])))

    dist = space.metric.dist(point, base_point)
    assert gs.isclose(dist, 1e-6, atol=0.0, rtol=1e-6)


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
