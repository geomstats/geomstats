r"""Unit tests for the space of PSD matrices of rank k."""

import geomstats.backend as gs
from geomstats.geometry.rank_k_psd_matrices import (
    BuresWassersteinBundle,
    PSDMatrices,
    PSDMetricBuresWasserstein,
)
from tests.conftest import Parametrizer
from tests.data.rank_k_psd_matrices_data import (
    BuresWassersteinBundleTestData,
    PSDMatricesTestData,
    TestDataPSDMetricBuresWasserstein,
)
from tests.geometry_test_cases import (
    FiberBundleTestCase,
    ManifoldTestCase,
    QuotientMetricTestCase,
)


class TestPSDMatrices(ManifoldTestCase, metaclass=Parametrizer):
    space = PSDMatrices

    testing_data = PSDMatricesTestData()

    def test_belongs(self, n, k, mat, expected):
        space = self.space(n, k)
        self.assertAllClose(space.belongs(gs.array(mat)), gs.array(expected))


class TestBuresWassersteinBundle(FiberBundleTestCase, metaclass=Parametrizer):
    bundle = BuresWassersteinBundle

    testing_data = BuresWassersteinBundleTestData()


class TestPSDMetricBuresWasserstein(QuotientMetricTestCase, metaclass=Parametrizer):

    space = PSDMatrices
    metric = connection = PSDMetricBuresWasserstein
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_log_after_exp = True
    skip_test_dist_is_smaller_than_bundle_dist = True
    skip_test_log_is_horizontal = True

    testing_data = TestDataPSDMetricBuresWasserstein()

    def test_exp(self, n, tangent_vec, base_point, expected):
        metric = PSDMetricBuresWasserstein(n, n)
        result = metric.exp(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_log(self, n, point, base_point, expected):
        metric = PSDMetricBuresWasserstein(n, n)
        result = metric.log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, expected)
