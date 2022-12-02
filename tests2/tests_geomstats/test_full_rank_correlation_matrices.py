import random

import pytest

from geomstats.geometry.full_rank_correlation_matrices import (
    FullRankCorrelationMatrices,
)
from geomstats.test.geometry.full_rank_correlation_matrices import (
    FullRankCorrelationMatricesTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.full_rank_correlation_matrices_data import (
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
