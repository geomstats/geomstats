import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.poisson import (
    PoissonDistributions,
    PoissonDistributionsRandomVariable,
)
from geomstats.test_cases.geometry.riemannian_metric import (
    CoordinateRiemannianMetricTestCase,
)
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


@pytest.mark.redundant
class TestPoissonMetric(
    CoordinateRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PoissonDistributions()
    testing_data = PoissonMetricTestData(
        space,
        excluded_methods=("sectional_curvature",),
        generator_kwargs={
            "amplitude": 5.0,
            "bounded_metric_norm": True,
            "within_injectivity_radius": True,
        },
    )
