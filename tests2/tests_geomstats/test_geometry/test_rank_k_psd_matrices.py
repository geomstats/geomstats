import random

import pytest

from geomstats.geometry.full_rank_matrices import FullRankMatrices
from geomstats.geometry.matrices import MatricesMetric
from geomstats.geometry.rank_k_psd_matrices import (
    BuresWassersteinBundle,
    PSDBuresWassersteinMetric,
    PSDMatrices,
)
from geomstats.test.geometry.rank_k_psd_matrices import (
    BuresWassersteinBundleTestCase,
    PSDBuresWassersteinMetricTestCase,
    RankKPSDMatricesTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.rank_k_psd_matrices_data import (
    BuresWassersteinBundleTestData,
    PSDBuresWassersteinMetricTestData,
    RankKPSDMatricesTestData,
)


def _get_random_params():
    while True:
        a = random.randint(2, 6)
        b = random.randint(2, 6)

        if a != b:
            break

    if a > b:
        n, k = a, b
    else:
        n, k = b, a

    return n, k


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
        (3, 2),
        _get_random_params(),
    ],
)
def spaces(request):
    if isinstance(request.param, int):
        n = k = request.param
    else:
        n, k = request.param
    request.cls.space = PSDMatrices(n=n, k=k, equip=False)


@pytest.mark.usefixtures("spaces")
class TestRankKPSDMatrices(RankKPSDMatricesTestCase, metaclass=DataBasedParametrizer):
    testing_data = RankKPSDMatricesTestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
        (3, 2),
        _get_random_params(),
    ],
)
def bundle_spaces(request):
    if isinstance(request.param, int):
        n = k = request.param
    else:
        n, k = request.param
    request.cls.base = PSDMatrices(n=n, k=k, equip=False)
    request.cls.total_space = total_space = FullRankMatrices(n=n, k=k, equip=False)
    total_space.equip_with_metric(MatricesMetric)
    request.cls.bundle = BuresWassersteinBundle(total_space)


@pytest.mark.usefixtures("bundle_spaces")
class TestBuresWassersteinBundle(
    BuresWassersteinBundleTestCase, metaclass=DataBasedParametrizer
):
    testing_data = BuresWassersteinBundleTestData()


@pytest.fixture(
    scope="class",
    params=[
        random.randint(2, 5),
        (3, 2),
        _get_random_params(),
    ],
)
def spaces_with_quotient_metric(request):
    if isinstance(request.param, int):
        n = k = request.param
    else:
        n, k = request.param
    space = request.cls.space = PSDMatrices(n=n, k=k, equip=False)
    space.equip_with_metric(PSDBuresWassersteinMetric)


@pytest.mark.usefixtures("spaces_with_quotient_metric")
class TestPSDBuresWassersteinMetric(
    PSDBuresWassersteinMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PSDBuresWassersteinMetricTestData()
