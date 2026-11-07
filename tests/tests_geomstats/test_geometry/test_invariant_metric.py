import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.invariant_metric import (
    BiInvariantMetric,
    _InvariantMetricMatrix,
    _InvariantMetricVector,
)
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test_cases.geometry.invariant_metric import (
    BiInvariantMetricTestCase,
    InvariantMetricMatrixTestCase,
    InvariantMetricVectorTestCase,
)

from .data.invariant_metric import (
    InvariantMetricMatrixSO3TestData,
    InvariantMetricMatrixTestData,
    InvariantMetricTestData,
)


@pytest.mark.smoke
class TestInvariantMetricMatrixSO3(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialOrthogonal(3, equip=False)
    space.equip_with_metric(_InvariantMetricMatrix, left=True)
    testing_data = InvariantMetricMatrixSO3TestData()


@pytest.fixture(scope="class", params=[True, False])
def equipped_SO_matrix_groups_left_right(request):
    space = request.cls.space = SpecialOrthogonal(random.randint(2, 3), equip=False)
    space.equip_with_metric(_InvariantMetricMatrix, left=request.param)

    request.cls.testing_data.space = space


@pytest.mark.slow
@pytest.mark.usefixtures("equipped_SO_matrix_groups_left_right")
class TestInvariantMetricMatrixSO(
    InvariantMetricMatrixTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = InvariantMetricMatrixTestData(
        excluded_methods=("injectivity_radius",),
        method_defaults={"exp_belongs": {"atol": 1e-6}},
        generator_kwargs={"amplitude": 10.0},
        sample_locally=True,
    )


# NB: `DataBasedParametrizer` skips vectorization tests, which are slow here
@pytest.mark.slow
class TestInvariantMetricMatrixSE(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialEuclidean(random.randint(2, 3), equip=False)
    space.equip_with_metric(_InvariantMetricMatrix, left=False)

    testing_data = InvariantMetricMatrixTestData(
        space,
        method_defaults={"exp_belongs": {"atol": 1e-6}},
        generator_kwargs={"amplitude": 10.0},
        sample_locally=True,
    )


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

    request.cls.testing_data.space = space


# NB: `DataBasedParametrizer` skips vectorization tests, which are slow here
@pytest.mark.slow
@pytest.mark.usefixtures("equipped_vector_groups_left_right")
class TestInvariantMetricVector(
    InvariantMetricVectorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = InvariantMetricTestData(
        excluded_methods=(
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "invariance",
        ),
        sample_locally=True,
    )


class TestBiInvariantMetricVectorsSO3(
    BiInvariantMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = SpecialOrthogonal(3, point_type="vector", equip=False)
    space.equip_with_metric(BiInvariantMetric)

    testing_data = InvariantMetricTestData(
        space,
        excluded_methods=(
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "invariance",
        ),
    )


class TestBiInvariantMetricMatrixSO3(
    BiInvariantMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = SpecialOrthogonal(3, equip=False)
    space.equip_with_metric(BiInvariantMetric)

    testing_data = InvariantMetricTestData(space)
