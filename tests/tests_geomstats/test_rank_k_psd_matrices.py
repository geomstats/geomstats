r"""Unit tests for the space of PSD matrices of rank k."""

from geomstats.geometry.rank_k_psd_matrices import PSDMatrices
from tests.conftest import Parametrizer
from tests.data.rank_k_psd_matrices_data import (
    BuresWassersteinBundleTestData,
    PSDMatricesTestData,
    PSDMetricBuresWassersteinTestData,
)
from tests.geometry_test_cases import (
    FiberBundleTestCase,
    ManifoldTestCase,
    QuotientMetricTestCase,
)


class TestPSDMatrices(ManifoldTestCase, metaclass=Parametrizer):

    testing_data = PSDMatricesTestData()

    def test_belongs(self, n, k, mat, expected):
        space = self.Space(n, k)
        self.assertAllClose(space.belongs(mat), expected)


class TestBuresWassersteinBundle(FiberBundleTestCase, metaclass=Parametrizer):

    testing_data = BuresWassersteinBundleTestData()
    Base = testing_data.Base


class TestPSDMetricBuresWasserstein(QuotientMetricTestCase, metaclass=Parametrizer):

    space = PSDMatrices
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_log_after_exp = True
    skip_test_dist_is_smaller_than_bundle_dist = True
    skip_test_log_is_horizontal = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = PSDMetricBuresWassersteinTestData()
    Metric = testing_data.Metric

    def test_exp(self, bundle, tangent_vec, base_point, expected):
        bundle.equip_with_metric(self.Metric)
        result = bundle.metric.exp(tangent_vec, base_point)
        self.assertAllClose(result, expected)

    def test_log(self, bundle, point, base_point, expected):
        bundle.equip_with_metric(self.Metric)
        result = bundle.metric.log(point, base_point)
        self.assertAllClose(result, expected)
