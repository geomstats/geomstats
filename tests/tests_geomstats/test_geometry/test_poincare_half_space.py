import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.poincare_half_space import (
    PoincareHalfSpace,
    PoincareHalfSpaceMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.poincare_half_space import PoincareHalfSpaceTestCase
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
    RiemannianMetricTestCase,
)
from geomstats.variants.geometry.poincare_half_space import (
    PullbackPoincareHalfSpaceMetric,
)

from .data.poincare_half_space import (
    PoincareHalfSpace2TestData,
    PoincareHalfSpaceMetric2TestData,
    PoincareHalfSpaceMetricTestData,
    PoincareHalfSpaceTestData,
    PullbackPoincareHalfSpaceMetricCmpTestData,
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


@pytest.mark.redundant
class TestPullbackPoincareHalfSpaceMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    _dim = random.randint(2, 4)
    space = PoincareHalfSpace(dim=_dim)
    other_space = PoincareHalfSpace(dim=_dim, equip=False)
    other_space.equip_with_metric(PullbackPoincareHalfSpaceMetric)

    testing_data = PullbackPoincareHalfSpaceMetricCmpTestData()


def test_nearby_vertical_distance():
    """Vertical distances agree with the log height ratio even for nearby points."""
    metric = PoincareHalfSpace(dim=2).metric
    point = gs.array([0.0, 1.0], dtype=gs.float64)
    other = gs.array([[0.0, 1.0], [0.0, 1.0 + 1e-9]], dtype=gs.float64)
    expected = gs.log(other[..., -1])
    assert gs.allclose(metric.dist(point, other), expected, atol=0.0, rtol=1e-10)
    assert gs.allclose(
        metric.squared_dist(point, other), expected**2, atol=0.0, rtol=1e-10
    )
