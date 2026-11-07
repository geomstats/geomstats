from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.geometric import (
    GeometricDistributions,
    GeometricDistributionsRandomVariable,
)
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.geometric import (
    GeometricDistributionsTestCase,
)

from .data.geometric import GeometricDistributionsTestData, GeometricMetricTestData


class TestGeometricDistributions(
    GeometricDistributionsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = GeometricDistributions(equip=False)
    random_variable = GeometricDistributionsRandomVariable(space)
    testing_data = GeometricDistributionsTestData(space)


class TestGeometricMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = GeometricDistributions()
    testing_data = GeometricMetricTestData(
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
        generator_kwargs={"amplitude": 10.0},
    )
