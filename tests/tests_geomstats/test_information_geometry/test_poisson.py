from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.poisson import (
    PoissonDistributions,
    PoissonDistributionsRandomVariable,
)
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.poisson import (
    PoissonDistributionsTestCase,
)

from .data.poisson import PoissonDistributionsTestData, PoissonMetricTestData


class TestPoissonDistributions(
    PoissonDistributionsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PoissonDistributions(equip=False)
    random_variable = PoissonDistributionsRandomVariable(space)
    testing_data = PoissonDistributionsTestData(space)


class TestPoissonMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PoissonDistributions()
    testing_data = PoissonMetricTestData(
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
        generator_kwargs={"amplitude": 5.0},
    )
