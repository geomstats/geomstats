import random

import pytest

from geomstats.geometry.invariant_metric import (
    BiInvariantMetric,
    _InvariantMetricMatrix,
    _InvariantMetricVector,
)
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.geometry.invariant_metric import (
    BiInvariantMetricTestCase,
    InvariantMetricMatrixTestCase,
    InvariantMetricVectorTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.invariant_metric_data import (
    BiInvariantMetricSO3VectorTestData,
    BiInvariantMetricSOMatrixTestData,
    InvariantMetricMatrixSOTestData,
    InvariantMetricVectorSOTestData,
)

# TODO: comparison of different point types?


@pytest.fixture(
    scope="class",
    params=[
        (SpecialOrthogonal(random.randint(2, 3), equip=False), True),
        (SpecialOrthogonal(random.randint(2, 3), equip=False), False),
    ],
)
def equipped_SO_matrix_groups_left_right(request):
    # TODO: different metric matrix at identity?
    space, left = request.param
    request.cls.space = space
    space.equip_with_metric(_InvariantMetricMatrix, left=left)


@pytest.mark.slow
@pytest.mark.usefixtures("equipped_SO_matrix_groups_left_right")
class TestInvariantMetricMatrixSO(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    testing_data = InvariantMetricMatrixSOTestData()


@pytest.fixture(
    scope="class",
    params=[
        (SpecialOrthogonal(3, point_type="vector", equip=False), True),
        (SpecialOrthogonal(3, point_type="vector", equip=False), False),
    ],
)
def equipped_SO3_vector_groups_left_right(request):
    space, left = request.param
    request.cls.space = space
    space.equip_with_metric(_InvariantMetricVector, left=left)


@pytest.mark.usefixtures("equipped_SO3_vector_groups_left_right")
class TestInvariantMetricVectorSO(
    InvariantMetricVectorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = InvariantMetricVectorSOTestData()


@pytest.fixture(
    scope="class",
    params=[
        SpecialOrthogonal(3, point_type="vector", equip=False),
    ],
)
def equipped_SO3_vector_groups(request):
    request.cls.space = space = request.param
    space.equip_with_metric(BiInvariantMetric)


@pytest.mark.usefixtures("equipped_SO3_vector_groups")
class TestBiInvariantMetricSO3Vector(
    BiInvariantMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = BiInvariantMetricSO3VectorTestData()


@pytest.fixture(
    scope="class",
    params=[
        SpecialOrthogonal(3, equip=False),
    ],
)
def equipped_SO_matrix_groups(request):
    request.cls.space = space = request.param
    space.equip_with_metric(BiInvariantMetric)


@pytest.mark.usefixtures("equipped_SO_matrix_groups")
class TestBiInvariantMetricSOMatrix(
    BiInvariantMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = BiInvariantMetricSOMatrixTestData()
