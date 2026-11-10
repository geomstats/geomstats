import random

import pytest
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
from .data.fiber_bundle import FiberBundleTestData
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


class TestGrassmannian(LevelSetTestCase, metaclass=GeometricDataBasedParametrizer):
    _n, _p = _get_random_params()
    space = Grassmannian(n=_n, p=_p, equip=False)

    testing_data = LevelSetTestData(space)


@pytest.mark.smoke
class TestGrassmannianCanonicalMetric32(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = Grassmannian(3, 2)
    testing_data = GrassmannianCanonicalMetric32TestData()


class TestGrassmannianCanonicalMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n, _p = _get_random_params()
    space = Grassmannian(n=_n, p=_p, equip=False).equip_with_metric(
        GrassmannianCanonicalMetric
    )

    testing_data = RiemannianMetricTestData(space)


class TestGrassmannianBundle(
    FiberBundleTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n, _p = _get_random_params()

    total_space = Stiefel(_n, _p, equip=False)

    total_space.equip_with_metric(StiefelCanonicalMetric)
    total_space.equip_with_group_action("right_orthogonal_action")
    total_space.equip_with_quotient()

    base_space = Grassmannian(_n, _p, equip=False)

    testing_data = FiberBundleTestData(
        total_space=total_space,
        base_space=base_space,
        excluded_methods=(
            "integrability_tensor",
            "integrability_tensor_derivative",
        ),
    ).with_redundancy(projection_source="submersion_lift")


class TestGrassmannianQuotientMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n, _p = _get_random_params()
    space = Grassmannian(_n, _p).equip_with_metric(
        ScalarProductMetric,
        scale=1 / 2.0,
    )

    _total_space = Stiefel(_n, _p)
    _total_space.equip_with_group_action("right_orthogonal_action")
    _total_space.equip_with_quotient()

    other_space = Grassmannian(_n, _p, equip=False).equip_with_metric(
        QuotientMetric, total_space=_total_space
    )

    testing_data = RiemannianMetricComparisonTestData(
        space,
        excluded_methods=(
            "injectivity_radius",
            "parallel_transport_ivp",
            "parallel_transport_bvp",
        )
    )
