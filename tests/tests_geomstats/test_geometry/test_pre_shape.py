import random

import pytest

from geomstats.geometry.pre_shape import PreShapeSpace
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
    k_landmarks, ambient_dim = request.param
    request.cls.space = PreShapeSpace(k_landmarks, ambient_dim)


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
    k_landmarks, ambient_dim = request.param

    request.cls.total_space = total_space = PreShapeSpace(k_landmarks, ambient_dim)

    total_space.equip_with_group_action("rotations")
    total_space.equip_with_quotient_structure()

    request.cls.base = total_space.quotient


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
    k_landmarks, ambient_dim = request.param

    request.cls.space = PreShapeSpace(k_landmarks, ambient_dim)


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
    k_landmarks, ambient_dim = request.param

    total_space = PreShapeSpace(k_landmarks, ambient_dim)

    total_space.equip_with_group_action("rotations")
    total_space.equip_with_quotient_structure()

    request.cls.total_space = total_space
    request.cls.space = total_space.quotient


@pytest.mark.usefixtures("spaces_with_quotient")
class TestKendallShapeMetric(
    KendallShapeMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = KendallShapeMetricTestData()


class TestKendallShapeMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    _k_landmarks = random.randint(4, 5)
    _ambient_dim = 2
    _total_space = PreShapeSpace(_k_landmarks, _ambient_dim)

    _total_space.equip_with_group_action("rotations")
    _total_space.equip_with_quotient_structure()

    space = _total_space.quotient

    other_total_space = PreShapeSpace(_k_landmarks, _ambient_dim)

    other_total_space.equip_with_group_action("rotations")
    other_total_space.equip_with_quotient_structure()

    other_space = other_total_space.quotient
    other_space.equip_with_metric(
        QuotientMetric,
        total_space=other_total_space,
    )

    testing_data = KendallShapeMetricCmpTestData()
