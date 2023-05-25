"""Unit tests for the complex Riemannian metrics."""

import tests.conftest
from tests.conftest import Parametrizer
from tests.data.complex_riemannian_metric_data import ComplexRiemannianMetricTestData
from tests.geometry_test_cases import RiemannianMetricTestCase


class TestComplexRiemannianMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_log = True
    skip_test_inner_coproduct = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = ComplexRiemannianMetricTestData()

    def test_cometric_matrix(self, metric, base_point, expected):
        result = metric.cometric_matrix(base_point)
        self.assertAllClose(result, expected)

    def test_inner_coproduct(
        self, metric, cotangent_vec_a, cotangent_vec_b, base_point, expected
    ):
        result = metric.inner_coproduct(cotangent_vec_a, cotangent_vec_b, base_point)
        self.assertAllClose(result, expected)

    def test_hamiltonian(self, metric, state, expected):
        result = metric.hamiltonian(state)
        self.assertAllClose(result, expected)

    @tests.conftest.torch_only
    def test_inner_product_derivative_matrix(self, metric, base_point, expected):
        result = metric.inner_product_derivative_matrix(base_point)
        self.assertAllClose(result, expected)

    def test_inner_product(
        self, metric, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        result = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(result, expected)

    def test_normalize(self, metric, tangent_vec, point, expected, atol):
        result = metric.norm(metric.normalize(tangent_vec, point), point)
        self.assertAllClose(result, expected, atol)

    def test_random_unit_tangent_vec(self, metric, point, n_vectors, expected, atol):
        result = metric.norm(metric.random_unit_tangent_vec(point, n_vectors), point)
        self.assertAllClose(result, expected, atol)

    @tests.conftest.torch_only
    def test_christoffels(self, metric, base_point, expected):
        result = metric.christoffels(base_point)
        self.assertAllClose(result, expected)

    @tests.conftest.torch_only
    def test_exp(self, metric, tangent_vec, base_point, expected):
        result = metric.exp(tangent_vec, base_point)
        self.assertAllClose(result, expected)
