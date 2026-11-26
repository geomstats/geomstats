import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.beta import (
    BetaDistributions,
    BetaDistributionsRandomVariable,
)
from geomstats.information_geometry.dirichlet import DirichletMetric
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.geometry.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestCase,
)
from geomstats.test_cases.information_geometry.beta import (
    BetaDistributionsTestCase,
    BetaMetricTestCase,
)
from geomstats.variants.information_geometry.beta import (
    BetaMetric as MetricMatrixBetaMetric,
)

from ..test_geometry.data.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestData,
)
from .data.beta import (
    BetaDistributionsTestData,
    BetaMetricTestData,
)


class TestBetaDistributions(
    BetaDistributionsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = BetaDistributions(equip=False)
    random_variable = BetaDistributionsRandomVariable(space)
    testing_data = BetaDistributionsTestData(space)


@pytest.mark.slow
class TestBetaMetric(BetaMetricTestCase, metaclass=GeometricDataBasedParametrizer):
    space = BetaDistributions()
    testing_data = BetaMetricTestData(
        space,
        excluded_methods=(
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
            "curvature_derivative",
            "directional_curvature_derivative",
        ),
        method_marks={
            "xfail": (
                "squared_dist_is_symmetric",
                "exp_after_log",
                "exp_diagonal_is_totally_geodesic",
                "log_after_exp",
            )
        },
        method_defaults={
            "exp_after_log": {"atol": 1e-3},
            "log_after_exp": {"atol": 1e-3},
            "squared_dist_is_symmetric": {"atol": 1e-2},
        },
        generator_kwargs={
            "amplitude": 5.0,
            "random_point_kwargs": {"low": 0.5, "bound": 4.5},
        },
        point_counts=[1, 2],
        sample_locally=True,
    )


@pytest.mark.redundant
class TestBetaMetricAsDirichletCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = BetaDistributions()
    other_space = BetaDistributions(equip=False).equip_with_metric(DirichletMetric)
    testing_data = CoordinateRiemannianMetricComparisonTestData(
        space,
        included_methods=(
            "metric_matrix",
            "christoffels",
            "riemann_tensor",
            "sectional_curvature",
            "scalar_curvature",
        ),
        generator_kwargs={"random_point_kwargs": {"low": 0.5, "bound": 4.5}},
    )


@autodiff_only
@pytest.mark.redundant
class TestBetaMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = BetaDistributions()
    other_space = BetaDistributions(equip=False).equip_with_metric(
        MetricMatrixBetaMetric
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
        generator_kwargs={"random_point_kwargs": {"low": 0.5, "bound": 4.5}},
    )
