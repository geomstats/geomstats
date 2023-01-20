import random

import pytest

from geomstats.geometry.full_rank_correlation_matrices import (
    CorrelationMatricesBundle,
    FullRankCorrelationMatrices,
)
from geomstats.test.geometry.full_rank_correlation_matrices import (
    CorrelationMatricesBundleTestCase,
    FullRankCorrelationMatricesTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.full_rank_correlation_matrices_data import (
    CorrelationMatricesBundleTestData,
    FullRankCorrelationMatricesTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        3,
        random.randint(4, 8),
    ],
)
def spaces(request):
    request.cls.space = FullRankCorrelationMatrices(n=request.param)


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
def bundle_spaces(request):
    n = request.param
    request.cls.space = CorrelationMatricesBundle(n=n)
    request.cls.base = FullRankCorrelationMatrices(n=n)


@pytest.mark.usefixtures("bundle_spaces")
class TestCorrelationMatricesBundle(
    CorrelationMatricesBundleTestCase, metaclass=DataBasedParametrizer
):
    testing_data = CorrelationMatricesBundleTestData()
