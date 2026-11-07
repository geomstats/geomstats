import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.information_geometry.dirichlet import DirichletDistributions
from geomstats.test_cases.information_geometry.dirichlet import (
    DirichletDistributionsTestCase,
    DirichletMetricTestCase,
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
        ),
        method_marks={
            "xfail": (
                "dist_is_symmetric",
                "squared_dist_is_symmetric",
                "exp_after_log",
                "exp_diagonal_is_totally_geodesic",
            )
        },
        method_defaults={"log_after_exp": {"atol": 1e-3}},
        generator_kwargs={"amplitude": 5.0},
    )
