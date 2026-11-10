import random

import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.test_cases.geometry.pre_shape import (
    KendallShapeMetricTestCase,
    PreShapeBundleTestCase,
    PreShapeSpaceTestCase,
)
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
    RiemannianMetricTestCase,
)

from .data.pre_shape import (
    KendallShapeMetricCmpTestData,
    KendallShapeMetricTestData,
    PreShapeBundleTestData,
    PreShapeSpaceTestData,
)
from .data.riemannian_metric import RiemannianMetricTestData


class TestPreShapeSpace(
    PreShapeSpaceTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PreShapeSpace(random.randint(3, 5), 2)
    testing_data = PreShapeSpaceTestData(space)


class TestPreShapeMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PreShapeSpace(random.randint(3, 5), 2)
    testing_data = RiemannianMetricTestData(space)


@pytest.fixture(
    scope="class",
    params=[
        (3, 2),
        (random.randint(4, 5), 2),
    ],
)
def bundles(request):
    k_landmarks, ambient_dim = request.param

    total_space = request.cls.total_space = PreShapeSpace(k_landmarks, ambient_dim)
    total_space.equip_with_group_action("rotations")
    total_space.equip_with_quotient()

    base_space = request.cls.base_space = total_space.quotient

    request.cls.testing_data.total_space = total_space
    request.cls.testing_data.base_space = base_space


@pytest.mark.usefixtures("bundles")
class TestPreShapeBundle(
    PreShapeBundleTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = PreShapeBundleTestData(
        excluded_methods=("tangent_riemannian_submersion_after_horizontal_lift",),
    ).with_redundancy(projection_source="vertical", custom_is_horizontal=True)


@pytest.fixture(
    scope="class",
    params=[
        (3, 2),
        (random.randint(4, 5), 2),
    ],
)
def spaces_with_quotient(request):
    k_landmarks, ambient_dim = request.param

    total_space = PreShapeSpace(k_landmarks, ambient_dim)
    total_space.equip_with_group_action("rotations")
    total_space.equip_with_quotient()

    space = total_space.quotient

    request.cls.total_space = total_space
    request.cls.space = space

    request.cls.testing_data.total_space = total_space
    request.cls.testing_data.space = space


@pytest.mark.usefixtures("spaces_with_quotient")
class TestKendallShapeMetric(
    KendallShapeMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = KendallShapeMetricTestData(
        excluded_methods=(
            "injectivity_radius",
            "sectional_curvature_lower_bound",
        ),
        method_defaults={
            "parallel_transport_bvp_transported_is_tangent": {"atol": 1e-4},
            "parallel_transport_ivp_transported_is_tangent": {"atol": 1e-4},
            "parallel_transport_ivp_transported_is_horizontal": {"atol": 1e-4},
            "curvature_bianchi_identity": {"atol": 1e-4},
        },
    )


class TestKendallShapeMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    _k_landmarks = random.randint(4, 5)
    _ambient_dim = 2
    _total_space = PreShapeSpace(_k_landmarks, _ambient_dim)

    _total_space.equip_with_group_action("rotations")
    _total_space.equip_with_quotient()

    space = _total_space.quotient

    _other_total_space = PreShapeSpace(_k_landmarks, _ambient_dim)

    _other_total_space.equip_with_group_action("rotations")
    _other_total_space.equip_with_quotient()

    other_space = _other_total_space.quotient
    other_space.equip_with_metric(
        QuotientMetric,
        total_space=_other_total_space,
    )

    testing_data = KendallShapeMetricCmpTestData(space)
