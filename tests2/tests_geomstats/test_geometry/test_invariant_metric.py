import random

import pytest

from geomstats.geometry.invariant_metric import (
    BiInvariantMetric,
    _InvariantMetricMatrix,
    _InvariantMetricVector,
)
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.invariant_metric import (
    BiInvariantMetricTestCase,
    InvariantMetricMatrixTestCase,
    InvariantMetricVectorTestCase,
)

from .data.invariant_metric import (
    BiInvariantMetricMatrixSOTestData,
    BiInvariantMetricVectorSO3TestData,
    InvariantMetricMatrixSETestData,
    InvariantMetricMatrixSOTestData,
    InvariantMetricVectorTestData,
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

    request.cls.data_generator = RandomDataGenerator(space, amplitude=5.0)


@pytest.mark.slow
@pytest.mark.usefixtures("equipped_SO_matrix_groups_left_right")
class TestInvariantMetricMatrixSO(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    testing_data = InvariantMetricMatrixSOTestData()


@pytest.fixture(
    scope="class",
    params=[
        SpecialEuclidean(random.randint(2, 3), equip=False),
    ],
)
def equipped_SE_matrix_groups(request):
    space = request.cls.space = request.param
    space.equip_with_metric(_InvariantMetricMatrix, left=False)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)


@pytest.mark.slow
@pytest.mark.usefixtures("equipped_SE_matrix_groups")
class TestInvariantMetricMatrixSE(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    testing_data = InvariantMetricMatrixSETestData()


@pytest.fixture(
    scope="class",
    params=[
        (SpecialOrthogonal(3, point_type="vector", equip=False), True),
        (SpecialOrthogonal(3, point_type="vector", equip=False), False),
        (SpecialEuclidean(2, point_type="vector", equip=False), True),
        (SpecialEuclidean(2, point_type="vector", equip=False), False),
        (SpecialEuclidean(3, point_type="vector", equip=False), True),
        (SpecialEuclidean(3, point_type="vector", equip=False), False),
    ],
)
def equipped_vector_groups_left_right(request):
    space, left = request.param
    request.cls.space = space
    space.equip_with_metric(_InvariantMetricVector, left=left)


@pytest.mark.slow
@pytest.mark.usefixtures("equipped_vector_groups_left_right")
class TestInvariantMetricVector(
    InvariantMetricVectorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = InvariantMetricVectorTestData()


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
class TestBiInvariantMetricVectorSO3(
    BiInvariantMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = BiInvariantMetricVectorSO3TestData()


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
class TestBiInvariantMetricMatrixSO(
    BiInvariantMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = BiInvariantMetricMatrixSOTestData()
