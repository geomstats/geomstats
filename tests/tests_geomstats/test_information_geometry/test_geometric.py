import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.geometric import (
    GeometricDistributions,
    GeometricDistributionsRandomVariable,
)
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.geometry.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestCase,
    CoordinateRiemannianMetricTestCase,
)
from geomstats.test_cases.information_geometry.geometric import (
    GeometricDistributionsTestCase,
)
from geomstats.variants.information_geometry.geometric import (
    GeometricMetric as MetricMatrixGeometricMetric,
)
from geomstats.variants.information_geometry.geometric import PullbackGeometricMetric

from ..test_geometry.data.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestData,
)
from .data.geometric import GeometricDistributionsTestData, GeometricMetricTestData


class TestGeometricDistributions(
    GeometricDistributionsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = GeometricDistributions(equip=False)
    random_variable = GeometricDistributionsRandomVariable(space)
    testing_data = GeometricDistributionsTestData(space)


class TestGeometricMetric(
    CoordinateRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = GeometricDistributions()
    testing_data = GeometricMetricTestData(
        space,
        excluded_methods=("sectional_curvature",),
        generator_kwargs={"within_injectivity_radius": True},
    )


@autodiff_only
@pytest.mark.redundant
class TestGeometricMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = GeometricDistributions()
    other_space = GeometricDistributions(equip=False).equip_with_metric(
        MetricMatrixGeometricMetric
    )
    testing_data = CoordinateRiemannianMetricComparisonTestData(
        space,
        included_methods=(
            "christoffels",
            "inner_product_derivative_matrix",
            "jacobian_christoffels",
            "curvature",
            "scalar_curvature",
        ),
    )


@autodiff_only
@pytest.mark.redundant
class TestPullbackGeometricMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = GeometricDistributions()
    other_space = GeometricDistributions(equip=False).equip_with_metric(
        PullbackGeometricMetric
    )
    testing_data = CoordinateRiemannianMetricComparisonTestData(
        space,
        included_methods=(
            "metric_matrix",
            "christoffels",
            "inner_product_derivative_matrix",
            "jacobian_christoffels",
            "curvature",
            "scalar_curvature",
        ),
    )
