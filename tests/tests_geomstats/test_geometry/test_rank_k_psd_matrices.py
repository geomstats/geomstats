import random

import pytest

from geomstats.geometry.full_rank_matrices import FullRankMatrices
from geomstats.geometry.matrices import MatricesMetric
from geomstats.geometry.rank_k_psd_matrices import (
    BuresWassersteinBundle,
    PSDBuresWassersteinMetric,
    PSDMatrices,
    RankKPSDMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RankKPSDMatricesRandomDataGenerator
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.mixins import ProjectionTestCaseMixins
from geomstats.test_cases.geometry.quotient_metric import QuotientMetricTestCase

from .data.rank_k_psd_matrices import (
    BuresWassersteinBundleTestData,
    PSD22BuresWassersteinMetricTestData,
    PSD33BuresWassersteinMetricTestData,
    PSDBuresWassersteinMetricTestData,
    RankKPSDMatrices32TestData,
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
        (3, 2),
        _get_random_params(),
    ],
)
def spaces(request):
    n, k = request.param
    space = request.cls.space = RankKPSDMatrices(n=n, k=k, equip=False)

    request.cls.data_generator = RankKPSDMatricesRandomDataGenerator(space)


@pytest.mark.usefixtures("spaces")
class TestRankKPSDMatrices(
    ProjectionTestCaseMixins, ManifoldTestCase, metaclass=DataBasedParametrizer
):
    testing_data = RankKPSDMatricesTestData()


@pytest.mark.smoke
class TestRankPSDMatrices32(ManifoldTestCase, metaclass=DataBasedParametrizer):
    space = RankKPSDMatrices(n=3, k=2, equip=False)
    testing_data = RankKPSDMatrices32TestData()


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
    total_space.fiber_bundle = BuresWassersteinBundle(total_space)


@pytest.mark.usefixtures("bundle_spaces")
class TestBuresWassersteinBundle(FiberBundleTestCase, metaclass=DataBasedParametrizer):
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
    QuotientMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PSDBuresWassersteinMetricTestData()


@pytest.mark.smoke
class TestPSD22BuresWassersteinMetric(
    QuotientMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PSDMatrices(n=2, k=2, equip=False)
    space.equip_with_metric(PSDBuresWassersteinMetric)

    testing_data = PSD22BuresWassersteinMetricTestData()


@pytest.mark.smoke
class TestPSD33BuresWassersteinMetric(
    QuotientMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PSDMatrices(n=3, k=3, equip=False)
    space.equip_with_metric(PSDBuresWassersteinMetric)

    testing_data = PSD33BuresWassersteinMetricTestData()
