import random

import pytest
from polpo.testing.composition import MarkedGeometricTestData
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.grassmannian import Grassmannian, GrassmannianCanonicalMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.geometry.stiefel import Stiefel, StiefelCanonicalMetric
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
    RiemannianMetricTestCase,
)

from .data.base import LevelSetTestData
from .data.fiber_bundle import (
    FiberBundleProjectionTestData,
    FiberBundleSubmersionLiftTestData,
    FiberBundleTestData,
)
from .data.grassmannian import (
    Grassmannian32TestData,
    GrassmannianCanonicalMetric32TestData,
)
from .data.riemannian_metric import (
    RiemannianMetricComparisonTestData,
    RiemannianMetricTestData,
)


def _get_random_params():
    while True:
        a = random.randint(2, 6)
        b = random.randint(2, 6)

        if a != b:
            break

    if a > b:
        n, p = a, b
    else:
        n, p = b, a

    return n, p


@pytest.mark.smoke
class TestGrassmannian32(LevelSetTestCase, metaclass=DataBasedParametrizer):
    space = Grassmannian(3, 2, equip=False)
    testing_data = Grassmannian32TestData()


@pytest.fixture(
    scope="class",
    params=[_get_random_params()],
)
def spaces(request):
    n, p = request.param
    space = request.cls.space = Grassmannian(n=n, p=p, equip=False)

    request.cls.testing_data.space = space


@pytest.mark.usefixtures("spaces")
class TestGrassmannian(LevelSetTestCase, metaclass=GeometricDataBasedParametrizer):
    testing_data = LevelSetTestData()


@pytest.mark.smoke
class TestGrassmannianCanonicalMetric32(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = Grassmannian(3, 2)
    testing_data = GrassmannianCanonicalMetric32TestData()


@pytest.fixture(
    scope="class",
    params=[_get_random_params()],
)
def equipped_spaces(request):
    n, p = request.param
    space = Grassmannian(n=n, p=p, equip=False)
    request.cls.space = space
    space.equip_with_metric(GrassmannianCanonicalMetric)

    request.cls.testing_data.space = space


@pytest.mark.usefixtures("equipped_spaces")
class TestGrassmannianCanonicalMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = RiemannianMetricTestData()


class TestGrassmannianBundle(
    FiberBundleTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n, _p = _get_random_params()

    total_space = Stiefel(_n, _p, equip=False)

    total_space.equip_with_metric(StiefelCanonicalMetric)
    total_space.equip_with_group_action("right_orthogonal_action")
    total_space.equip_with_quotient()

    base_space = Grassmannian(_n, _p, equip=False)

    testing_data = (
        FiberBundleTestData(
            excluded_methods=(
                "integrability_tensor",
                "integrability_tensor_derivative",
            )
        )
        + FiberBundleSubmersionLiftTestData()
        + MarkedGeometricTestData(
            FiberBundleProjectionTestData(), marks=(pytest.mark.redundant,)
        )
    )

    testing_data.propagate("total_space", total_space)
    testing_data.propagate("base_space", base_space)


@pytest.fixture(
    scope="class",
    params=[_get_random_params()],
)
def grassmannian_with_quotient_metric(request):
    n, p = request.param
    space = request.cls.space = Grassmannian(n, p).equip_with_metric(
        ScalarProductMetric,
        scale=1 / 2.0,
    )

    request.cls.other_space = other_space = Grassmannian(n, p, equip=False)

    total_space = Stiefel(n, p)
    total_space.equip_with_group_action("right_orthogonal_action")
    total_space.equip_with_quotient()

    other_space.equip_with_metric(QuotientMetric, total_space=total_space)

    request.cls.testing_data.space = space


@pytest.mark.usefixtures("grassmannian_with_quotient_metric")
class TestGrassmannianQuotientMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = RiemannianMetricComparisonTestData(
        excluded_methods=(
            "injectivity_radius",
            "parallel_transport_ivp",
            "parallel_transport_bvp",
        )
    )
