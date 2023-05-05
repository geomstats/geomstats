r"""Unit tests for the space of PSD matrices of rank k."""

from tests.conftest import Parametrizer
from tests.data.rank_k_psd_matrices_data import (
    BuresWassersteinBundleTestData,
    PSDBuresWassersteinMetricTestData,
    PSDMatricesTestData,
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
    TotalSpace = testing_data.TotalSpace
    Bundle = testing_data.Bundle


class TestPSDBuresWassersteinMetric(QuotientMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_log_after_exp = True
    skip_test_log_is_horizontal = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = PSDBuresWassersteinMetricTestData()
    Metric = testing_data.Metric

    def test_inner_product(
        self, space, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        space.equip_with_metric(self.Metric)
        result = space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(result, expected)

    def test_exp(self, space, tangent_vec, base_point, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(result, expected)

    def test_log(self, space, point, base_point, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.log(point, base_point)
        self.assertAllClose(result, expected)
