import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.gamma import (
    GammaDistributions,
    GammaDistributionsRandomVariable,
    ShapeRateGammaDistributions,
)
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.geometry.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestCase,
    CoordinateRiemannianMetricTestCase,
)
from geomstats.test_cases.information_geometry.gamma import GammaDistributionsTestCase
from geomstats.variants.information_geometry.gamma import (
    GammaMetric as MetricMatrixGammaMetric,
)

from ..test_geometry.data.pullback_metric import (
    get_coordinate_pullback_excluded_methods,
)
from ..test_geometry.data.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestData,
    CoordinateRiemannianMetricTestData,
)
from .data.gamma import (
    GammaDistributionsTestData,
    ShapeRateGammaMetricTestData,
)


class TestGammaDistributions(
    GammaDistributionsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = GammaDistributions(equip=False)
    random_variable = GammaDistributionsRandomVariable(space)
    testing_data = GammaDistributionsTestData(space)


class TestGammaMetric(
    CoordinateRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = GammaDistributions()
    testing_data = CoordinateRiemannianMetricTestData(
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
                "log_after_exp",
                "exp_after_log",
            )
        },
        method_defaults={
            "exp_after_log": {"atol": 1e-3},
            "log_after_exp": {"atol": 1e-3},
            "squared_dist_is_symmetric": {"atol": 1e-2},
        },
        generator_kwargs={
            "amplitude": 10.0,
            "random_point_kwargs": {"low": 0.5, "bound": 4.5},
        },
        sample_locally=True,
    )


@autodiff_only
@pytest.mark.redundant
class TestGammaMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = GammaDistributions()
    other_space = GammaDistributions(equip=False).equip_with_metric(
        MetricMatrixGammaMetric
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


@pytest.mark.redundant
class TestShapeRateGammaMetric(
    CoordinateRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ShapeRateGammaDistributions()
    testing_data = ShapeRateGammaMetricTestData(
        space,
        excluded_methods=get_coordinate_pullback_excluded_methods()
        + (
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
                "log_after_exp",
                "exp_after_log",
            )
        },
        method_defaults={
            "exp_after_log": {"atol": 1e-3},
            "log_after_exp": {"atol": 1e-3},
            "squared_dist_is_symmetric": {"atol": 1e-2},
        },
        generator_kwargs={
            "amplitude": 10.0,
            "random_point_kwargs": {"low": 0.5, "bound": 4.5},
        },
        sample_locally=True,
    )
