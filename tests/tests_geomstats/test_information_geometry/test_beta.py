import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.beta import (
    BetaDistributions,
    BetaDistributionsRandomVariable,
)
from geomstats.information_geometry.dirichlet import DirichletMetric
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
)
from geomstats.test_cases.information_geometry.beta import (
    BetaDistributionsTestCase,
    BetaMetricTestCase,
)

from .data.beta import (
    BetaDistributionsTestData,
    BetaMetricAsDirichletCmpTestData,
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
        ),
        method_marks={
            "xfail": (
                "dist_is_symmetric",
                "squared_dist_is_symmetric",
                "exp_after_log",
                "exp_diagonal_is_totally_geodesic",
                "log_after_exp",
            )
        },
        method_defaults={"log_after_exp": {"atol": 1e-2}},
        generator_kwargs={"amplitude": 5.0},
        point_counts=[1, 2],
    )


@pytest.mark.redundant
class TestBetaMetricAsDirichletCmp(
    RiemannianMetricComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = BetaDistributions()
    other_space = BetaDistributions(equip=False).equip_with_metric(DirichletMetric)
    testing_data = BetaMetricAsDirichletCmpTestData(space)
