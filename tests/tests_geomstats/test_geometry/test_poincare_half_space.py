import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.poincare_half_space import (
    PoincareHalfSpace,
    PoincareHalfSpaceMetric,
)
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.poincare_half_space import (
    PoincareHalfSpaceMetricTestCase,
)
from geomstats.test_cases.geometry.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestCase,
)
from geomstats.variants.geometry.poincare_half_space import (
    PullbackPoincareHalfSpaceMetric,
)

from .data.poincare_half_space import (
    PoincareHalfSpace2TestData,
    PoincareHalfSpaceMetric2TestData,
    PoincareHalfSpaceTestData,
)
from .data.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestData,
    CoordinateRiemannianMetricTestData,
)


class TestPoincareHalfSpace(
    VectorSpaceOpenSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PoincareHalfSpace(dim=random.randint(2, 5), equip=False)
    testing_data = PoincareHalfSpaceTestData(space)


@pytest.mark.smoke
class TestPoincareHalfSpace2(
    VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = PoincareHalfSpace(dim=2, equip=False)
    testing_data = PoincareHalfSpace2TestData()


class TestPoincareHalfSpaceMetric(
    PoincareHalfSpaceMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PoincareHalfSpace(dim=random.randint(2, 5))
    testing_data = CoordinateRiemannianMetricTestData(
        space,
        generator_kwargs={"bounded_metric_norm": True},
    )


@pytest.mark.smoke
class TestPoincareHalfSpaceMetric2(
    PoincareHalfSpaceMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PoincareHalfSpace(dim=2, equip=False)
    space.equip_with_metric(PoincareHalfSpaceMetric)
    testing_data = PoincareHalfSpaceMetric2TestData()


@autodiff_only
@pytest.mark.redundant
class TestPullbackPoincareHalfSpaceMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _dim = random.randint(2, 4)
    space = PoincareHalfSpace(dim=_dim)
    other_space = PoincareHalfSpace(dim=_dim, equip=False).equip_with_metric(
        PullbackPoincareHalfSpaceMetric
    )
    testing_data = CoordinateRiemannianMetricComparisonTestData(
        space,
        included_methods=(
            "metric_matrix",
            "christoffels",
            "inner_product_derivative_matrix",
            "jacobian_christoffels",
            "riemann_tensor",
            "curvature",
            "curvature_derivative",
            "sectional_curvature",
            "scalar_curvature",
            "squared_dist",
            "dist",
        ),
        generator_kwargs={"bounded_metric_norm": True},
    )
