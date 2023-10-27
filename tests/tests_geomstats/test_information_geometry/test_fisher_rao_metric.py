import pytest

from geomstats.information_geometry.beta import BetaDistributions
from geomstats.information_geometry.exponential import ExponentialDistributions
from geomstats.information_geometry.fisher_rao_metric import FisherRaoMetric
from geomstats.information_geometry.gamma import GammaDistributions
from geomstats.information_geometry.normal import UnivariateNormalDistributions
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
)

from .data.fisher_rao_metric import (
    FisherRaoMetricCmpBetaTestData,
    FisherRaoMetricCmpExponentialTestData,
    FisherRaoMetricCmpGammaTestData,
    FisherRaoMetricCmpUnivariateNormalTestData,
)


@pytest.mark.smoke
class TestFisherRaoCmpUnivariateNormal(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    support = (-20, 20)
    space = UnivariateNormalDistributions(equip=False)
    space.equip_with_metric(FisherRaoMetric, support=support)

    other_space = UnivariateNormalDistributions()

    testing_data = FisherRaoMetricCmpUnivariateNormalTestData()


@pytest.mark.smoke
class TestFisherRaoCmpExponential(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    support = (0, 100)
    space = ExponentialDistributions(equip=False)
    space.equip_with_metric(FisherRaoMetric, support=support)

    other_space = ExponentialDistributions()

    testing_data = FisherRaoMetricCmpExponentialTestData()


@pytest.mark.smoke
class TestFisherRaoCmpGamma(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    support = (0, 100)
    space = GammaDistributions(equip=False)
    space.equip_with_metric(FisherRaoMetric, support=support)

    other_space = GammaDistributions()

    testing_data = FisherRaoMetricCmpGammaTestData()


@pytest.mark.smoke
class TestFisherRaoCmpBeta(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    support = (0, 1)
    space = BetaDistributions(equip=False)
    space.equip_with_metric(FisherRaoMetric, support=support)

    other_space = BetaDistributions()

    testing_data = FisherRaoMetricCmpBetaTestData()
