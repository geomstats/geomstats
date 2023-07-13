import random

import pytest

from geomstats.geometry.poincare_half_space import (
    PoincareHalfSpace,
    PoincareHalfSpaceMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from tests2.tests_geomstats.test_geometry.data.poincare_half_space import (
    PoincareHalfSpaceMetricTestData,
)

from .data.poincare_half_space import (
    PoincareHalfSpace2TestData,
    PoincareHalfSpaceMetric2TestData,
    PoincareHalfSpaceTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = PoincareHalfSpace(dim=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestPoincareHalfSpace(OpenSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = PoincareHalfSpaceTestData()


@pytest.mark.smoke
class TestPoincareHalfSpace2(OpenSetTestCase, metaclass=DataBasedParametrizer):
    space = PoincareHalfSpace(dim=2, equip=False)
    testing_data = PoincareHalfSpace2TestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_spaces(request):
    space = request.cls.space = PoincareHalfSpace(dim=request.param, equip=False)
    space.equip_with_metric(PoincareHalfSpaceMetric)


@pytest.mark.usefixtures("equipped_spaces")
class TestPoincareHalfSpaceMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PoincareHalfSpaceMetricTestData()


@pytest.mark.smoke
class TestPoincareHalfSpaceMetric2(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PoincareHalfSpace(dim=2)
    testing_data = PoincareHalfSpaceMetric2TestData()
