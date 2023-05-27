import pytest

from geomstats.information_geometry.beta import BetaDistributions
from geomstats.information_geometry.exponential import ExponentialDistributions
from geomstats.information_geometry.fisher_rao_metric import FisherRaoMetric
from geomstats.information_geometry.gamma import GammaDistributions
from geomstats.information_geometry.normal import UnivariateNormalDistributions
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
)

from .data.fisher_rao_metric import FisherRaoMetricCmpTestData


@pytest.fixture(
    scope="class",
    params=[
        (
            UnivariateNormalDistributions(equip=False),
            UnivariateNormalDistributions(),
            (-10, 10),
        ),
        (
            GammaDistributions(equip=False),
            GammaDistributions(),
            (0, 10),
        ),
        (
            ExponentialDistributions(equip=False),
            ExponentialDistributions(),
            (0, 100),
        ),
        (
            BetaDistributions(equip=False),
            BetaDistributions(),
            (0, 1),
        ),
    ],
)
def spaces(request):
    space, other_space, support = request.param

    space.equip_with_metric(FisherRaoMetric, support=support)

    request.cls.space = space
    request.cls.other_space = other_space

    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)


@pytest.mark.usefixtures("spaces")
class TestFisherRaoMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    testing_data = FisherRaoMetricCmpTestData()
