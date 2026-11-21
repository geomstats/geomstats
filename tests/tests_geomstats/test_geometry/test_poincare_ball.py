import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.poincare_ball import PoincareBall, PoincareBallMetric
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.geometry.poincare_ball import (
    PoincareBallMetricTestCase,
    PoincareBallTestCase,
)
from geomstats.test_cases.geometry.retraction import RetractionTestCase
from geomstats.test_cases.geometry.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestCase,
)
from geomstats.variants.geometry.poincare_ball import (
    PoincareBallMetric as MetricMatrixPoincareBallMetric,
)

from .data.poincare_ball import (
    PoincareBall2MetricTestData,
    PoincareBall2TestData,
    PoincareBallMetricTestData,
    PoincareBallTestData,
)
from .data.retraction import RetractionTestData
from .data.riemannian_metric import CoordinateRiemannianMetricComparisonTestData


@pytest.mark.smoke
class TestPoincareBall2(PoincareBallTestCase, metaclass=DataBasedParametrizer):
    space = PoincareBall(dim=2, equip=False)
    testing_data = PoincareBall2TestData()


class TestPoincareBall(PoincareBallTestCase, metaclass=GeometricDataBasedParametrizer):
    space = PoincareBall(dim=random.randint(2, 5), equip=False)
    testing_data = PoincareBallTestData(space)


@pytest.mark.smoke
class TestPoincareBall2Metric(
    PoincareBallMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PoincareBall(dim=2, equip=False)
    space.equip_with_metric(PoincareBallMetric)
    testing_data = PoincareBall2MetricTestData()


class TestPoincareBallMetric(
    PoincareBallMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PoincareBall(dim=random.randint(2, 5))
    testing_data = PoincareBallMetricTestData(space)


@autodiff_only
@pytest.mark.redundant
class TestPoincareBallMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _dim = random.randint(2, 5)
    space = PoincareBall(dim=_dim)
    other_space = PoincareBall(dim=_dim, equip=False).equip_with_metric(
        MetricMatrixPoincareBallMetric
    )
    testing_data = CoordinateRiemannianMetricComparisonTestData(
        space,
        included_methods=(
            "christoffels",
            "inner_product_derivative_matrix",
            "jacobian_christoffels",
            "riemann_tensor",
            "scalar_curvature",
            "sectional_curvature",
        ),
    )


class TestPoincareBallRetraction(
    RetractionTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PoincareBall(dim=random.randint(2, 5))
    testing_data = RetractionTestData(
        space,
        excluded_methods=(
            "lifting",
            "lifting_is_tangent",
            "lifting_after_retraction",
            "retraction_after_lifting",
        ),
    )
