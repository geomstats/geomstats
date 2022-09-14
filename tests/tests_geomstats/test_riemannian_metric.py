"""Unit tests for the Riemannian metrics."""

import geomstats.tests
from tests.conftest import Parametrizer
from tests.data.riemannian_metric_data import RiemannianMetricTestData
from tests.geometry_test_cases import TestCase


class TestRiemannianMetric(TestCase, metaclass=Parametrizer):

    testing_data = RiemannianMetricTestData()

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

    @geomstats.tests.autograd_and_torch_only
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

    @geomstats.tests.autograd_and_torch_only
    def test_christoffels(self, metric, base_point, expected):
        result = metric.christoffels(base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_and_torch_only
    def test_exp(self, metric, tangent_vec, base_point, expected):
        result = metric.exp(tangent_vec, base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_and_torch_only
    def test_log(self, metric, point, base_point, expected):
        result = metric.log(point, base_point)
        self.assertAllClose(result, expected)
