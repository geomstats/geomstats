from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.exponential import (
    ExponentialDistributions,
    ExponentialDistributionsRandomVariable,
)
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.exponential import (
    ExponentialDistributionsTestCase,
)

from .data.exponential import (
    ExponentialDistributionsTestData,
    ExponentialMetricTestData,
)


class TestExponentialDistributions(
    ExponentialDistributionsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ExponentialDistributions(equip=False)
    random_variable = ExponentialDistributionsRandomVariable(space)
    testing_data = ExponentialDistributionsTestData(space)


class TestExponentialMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ExponentialDistributions()
    testing_data = ExponentialMetricTestData(
        space,
        excluded_methods=(
            "injectivity_radius",
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
        ),
    )
