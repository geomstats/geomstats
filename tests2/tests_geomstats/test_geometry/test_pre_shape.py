import random

import pytest

from geomstats.geometry.pre_shape import (
    PreShapeMetric,
    PreShapeSpace,
    PreShapeSpaceBundle,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.pre_shape import (
    KendallShapeMetricTestCase,
    PreShapeSpaceBundleTestCase,
    PreShapeSpaceTestCase,
)
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.pre_shape import (
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

    request.cls.space = space = PreShapeSpace(k_landmarks, m_ambient, equip=False)
    space.equip_with_metric(PreShapeMetric)


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

    space = PreShapeSpace(k_landmarks, m_ambient, equip=True)

    space.equip_with_group_action("rotations")
    space.equip_with_quotient_structure()

    request.cls.space = space.quotient


@pytest.mark.usefixtures("spaces_with_quotient")
class TestKendallShapeMetric(
    KendallShapeMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = KendallShapeMetricTestData()
