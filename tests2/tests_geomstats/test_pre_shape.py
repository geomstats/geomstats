import random

import pytest

from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.test.geometry.pre_shape import (
    PreShapeSpaceBundleTestCase,
    PreShapeSpaceTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.pre_shape_data import (
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
    space = PreShapeSpace(k_landmarks, m_ambient, equip=False)
    request.cls.base = space
    request.cls.space = space.fiber_bundle


@pytest.mark.usefixtures("bundles")
class TestPreShapeSpaceBundle(
    PreShapeSpaceBundleTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PreShapeSpaceBundleTestData()
