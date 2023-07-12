import random

import pytest

from geomstats.geometry.full_rank_correlation_matrices import (
    CorrelationMatricesBundle,
    FullRankCorrelationAffineQuotientMetric,
    FullRankCorrelationEuclideanCholeskyMetric,
    FullRankCorrelationMatrices,
)
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.full_rank_correlation_matrices import (
    CorrelationMatricesBundleTestCase,
    FullRankCorrelationAffineQuotientMetricTestCase,
    FullRankCorrelationMatricesTestCase,
)
from geomstats.test_cases.geometry.quotient_metric import QuotientMetricTestCase

from .data.full_rank_correlation_matrices import (
    CorrelationMatricesBundleTestData,
    FullRankCorrelationAffineQuotientMetricTestData,
    FullRankCorrelationMatricesTestData,
    FullRankEuclideanCholeskyMetricTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        3,
        random.randint(4, 8),
    ],
)
def spaces(request):
    request.cls.space = FullRankCorrelationMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestFullRankCorrelationMatrices(
    FullRankCorrelationMatricesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = FullRankCorrelationMatricesTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def bundles(request):
    n = request.param
    request.cls.total_space = total_space = SPDMatrices(n=n, equip=True)
    request.cls.bundle = CorrelationMatricesBundle(total_space)
    request.cls.base = FullRankCorrelationMatrices(n=n, equip=False)


@pytest.mark.usefixtures("bundles")
class TestCorrelationMatricesBundle(
    CorrelationMatricesBundleTestCase, metaclass=DataBasedParametrizer
):
    testing_data = CorrelationMatricesBundleTestData()


@pytest.fixture(
    scope="class",
    params=[
        3,
        random.randint(4, 5),
    ],
)
def affine_quotient_equipped_spaces(request):
    n = request.param
    request.cls.space = space = FullRankCorrelationMatrices(n=n, equip=False)
    space.equip_with_metric(FullRankCorrelationAffineQuotientMetric)


@pytest.mark.usefixtures("affine_quotient_equipped_spaces")
class TestFullRankCorrelationAffineQuotientMetric(
    FullRankCorrelationAffineQuotientMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = FullRankCorrelationAffineQuotientMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        3,
        random.randint(4, 5),
    ],
)
def euclidean_cholesky_equipped_spaces(request):
    n = request.param
    request.cls.space = space = FullRankCorrelationMatrices(n=n, equip=False)
    space.equip_with_metric(FullRankCorrelationEuclideanCholeskyMetric)


@pytest.mark.usefixtures("euclidean_cholesky_equipped_spaces")
class TestFullRankEuclideanCholeskyMetric(
    QuotientMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = FullRankEuclideanCholeskyMetricTestData()
