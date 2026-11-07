import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.information_geometry.dirichlet import DirichletDistributions
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.geometry.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestCase,
)
from geomstats.test_cases.information_geometry.dirichlet import (
    DirichletDistributionsTestCase,
    DirichletMetricTestCase,
)
from geomstats.variants.information_geometry.dirichlet import (
    DirichletMetric as MetricMatrixDirichletMetric,
)

from ..test_geometry.data.riemannian_metric import (
    CoordinateRiemannianMetricComparisonTestData,
)
from .data.dirichlet import (
    DirichletDistributions3TestData,
    DirichletDistributionsTestData,
    DirichletMetricTestData,
)


@pytest.mark.smoke
class TestDirichletDistributions3(
    DirichletDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = DirichletDistributions(dim=3, equip=False)
    testing_data = DirichletDistributions3TestData()


class TestDirichletDistributions(
    DirichletDistributionsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = DirichletDistributions(dim=random.randint(3, 4), equip=False)
    testing_data = DirichletDistributionsTestData(space)


@pytest.mark.slow
class TestDirichletMetric(
    DirichletMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = DirichletDistributions(dim=random.randint(3, 4))
    testing_data = DirichletMetricTestData(
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
        sample_locally=True,
    )


@autodiff_only
@pytest.mark.redundant
class TestDirichletMetricCmp(
    CoordinateRiemannianMetricComparisonTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _dim = random.randint(3, 4)
    space = DirichletDistributions(dim=_dim)
    other_space = DirichletDistributions(dim=_dim, equip=False).equip_with_metric(
        MetricMatrixDirichletMetric
    )
    testing_data = CoordinateRiemannianMetricComparisonTestData(
        space,
        included_methods=(
            "christoffels",
            "inner_product_derivative_matrix",
            "jacobian_christoffels",
        ),
        generator_kwargs={"random_point_kwargs": {"low": 0.5, "bound": 4.5}},
    )
