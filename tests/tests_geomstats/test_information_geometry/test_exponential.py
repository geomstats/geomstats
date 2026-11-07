import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.information_geometry.exponential import (
    ExponentialDistributions,
    ExponentialDistributionsRandomVariable,
)
from geomstats.test_cases.geometry.riemannian_metric import (
    CoordinateRiemannianMetricTestCase,
)
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


@pytest.mark.redundant
class TestExponentialMetric(
    CoordinateRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = ExponentialDistributions()
    testing_data = ExponentialMetricTestData(
        space,
        excluded_methods=("sectional_curvature",),
        generator_kwargs={"bounded_metric_norm": True},
    )
