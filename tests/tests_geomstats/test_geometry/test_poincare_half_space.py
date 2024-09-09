import random

import pytest

from geomstats.geometry.poincare_half_space import (
    PoincareHalfSpace,
    PoincareHalfSpaceMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.poincare_half_space import PoincareHalfSpaceTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.poincare_half_space import (
    PoincareHalfSpace2TestData,
    PoincareHalfSpaceMetric2TestData,
    PoincareHalfSpaceMetricTestData,
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
class TestPoincareHalfSpace(PoincareHalfSpaceTestCase, metaclass=DataBasedParametrizer):
    testing_data = PoincareHalfSpaceTestData()


@pytest.mark.smoke
class TestPoincareHalfSpace2(
    PoincareHalfSpaceTestCase, metaclass=DataBasedParametrizer
):
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
    request.cls.space = PoincareHalfSpace(dim=request.param)


@pytest.mark.usefixtures("equipped_spaces")
class TestPoincareHalfSpaceMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PoincareHalfSpaceMetricTestData()


@pytest.mark.smoke
class TestPoincareHalfSpaceMetric2(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PoincareHalfSpace(dim=2, equip=False)
    space.equip_with_metric(PoincareHalfSpaceMetric)
    testing_data = PoincareHalfSpaceMetric2TestData()

    def test_exp_and_coordinates_tangent(self, tangent_vec, base_point):
        end_point = self.space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(base_point[0], end_point[0])
