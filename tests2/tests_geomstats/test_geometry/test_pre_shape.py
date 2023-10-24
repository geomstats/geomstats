import random

import pytest

from geomstats.geometry.pre_shape import PreShapeSpace, PreShapeSpaceBundle
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.pre_shape import (
    KendallShapeMetricTestCase,
    PreShapeSpaceBundleTestCase,
    PreShapeSpaceTestCase,
)
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
    RiemannianMetricTestCase,
)

from .data.pre_shape import (
    KendallShapeMetricCmpTestData,
    KendallShapeMetricTestData,
    PreShapeMetricTestData,
    PreShapeSpaceBundleTestData,
    PreShapeSpaceTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        (3, 2),
        (random.randint(4, 5), 2),
    ],
)
def spaces(request):
    k_landmarks, m_ambient = request.param
    request.cls.space = PreShapeSpace(k_landmarks, m_ambient)


@pytest.mark.usefixtures("spaces")
class TestPreShapeSpace(PreShapeSpaceTestCase, metaclass=DataBasedParametrizer):
    testing_data = PreShapeSpaceTestData()


@pytest.fixture(
    scope="class",
    params=[
        (3, 2),
        (random.randint(4, 5), 2),
    ],
)
def bundles(request):
    k_landmarks, m_ambient = request.param

    space = PreShapeSpace(k_landmarks, m_ambient, equip=True)
    request.cls.total_space = request.cls.base = space

    request.cls.bundle = PreShapeSpaceBundle(space)


@pytest.mark.usefixtures("bundles")
class TestPreShapeSpaceBundle(
    PreShapeSpaceBundleTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PreShapeSpaceBundleTestData()


@pytest.fixture(
    scope="class",
    params=[
        (3, 2),
        (random.randint(4, 5), 2),
    ],
)
def equipped_spaces(request):
    k_landmarks, m_ambient = request.param

    request.cls.space = PreShapeSpace(k_landmarks, m_ambient)


@pytest.mark.usefixtures("spaces")
class TestPreShapeMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = PreShapeMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        (3, 2),
        (random.randint(4, 5), 2),
    ],
)
def spaces_with_quotient(request):
    k_landmarks, m_ambient = request.param

    space = PreShapeSpace(k_landmarks, m_ambient)

    space.equip_with_group_action("rotations")
    space.equip_with_quotient_structure()

    request.cls.space = space.quotient


@pytest.mark.usefixtures("spaces_with_quotient")
class TestKendallShapeMetric(
    KendallShapeMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = KendallShapeMetricTestData()


class TestKendallShapeMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    k_landmarks = random.randint(4, 5)
    m_ambient = 2
    total_space = PreShapeSpace(k_landmarks, m_ambient)

    total_space.equip_with_group_action("rotations")
    total_space.equip_with_quotient_structure()

    space = total_space.quotient

    other_total_space = PreShapeSpace(k_landmarks, m_ambient)

    other_total_space.equip_with_group_action("rotations")
    other_total_space.equip_with_quotient_structure()

    other_space = other_total_space.quotient
    other_space.equip_with_metric(
        QuotientMetric,
        fiber_bundle=other_space.metric.fiber_bundle,
    )

    testing_data = KendallShapeMetricCmpTestData()
