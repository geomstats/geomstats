import random

import pytest

from geomstats.information_geometry.binomial import (
    BinomialDistributions,
    BinomialDistributionsRandomVariable,
    BinomialMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.binomial import (
    BinomialDistributionsTestCase,
)

from .data.binomial import (
    Binomial5MetricTestData,
    Binomial7MetricTestData,
    Binomial10MetricTestData,
    BinomialDistributionsTestData,
    BinomialMetricTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    n_draws = request.param
    space = request.cls.space = BinomialDistributions(n_draws=n_draws, equip=False)
    request.cls.random_variable = BinomialDistributionsRandomVariable(space)


@pytest.mark.usefixtures("spaces")
class TestBinomialDistributions(
    BinomialDistributionsTestCase, metaclass=DataBasedParametrizer
):
    testing_data = BinomialDistributionsTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_spaces(request):
    request.cls.space = BinomialDistributions(n_draws=request.param)


@pytest.mark.usefixtures("equipped_spaces")
class TestBinomialMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = BinomialMetricTestData()


@pytest.mark.smoke
class TestBinomial5Metric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = BinomialDistributions(n_draws=5, equip=False)
    space.equip_with_metric(BinomialMetric)
    testing_data = Binomial5MetricTestData()


@pytest.mark.smoke
class TestBinomial7Metric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = BinomialDistributions(n_draws=7, equip=False)
    space.equip_with_metric(BinomialMetric)
    testing_data = Binomial7MetricTestData()


@pytest.mark.smoke
class TestBinomial10Metric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = BinomialDistributions(n_draws=10, equip=False)
    space.equip_with_metric(BinomialMetric)
    testing_data = Binomial10MetricTestData()
