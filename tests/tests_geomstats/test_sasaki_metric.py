"""Unit tests for the Sasaki metric."""

import tests.conftest
from tests.conftest import Parametrizer
from tests.data.sasaki_metric_data import SasakiMetricTestData
from tests.geometry_test_cases import TestCase


class TestSasakiMetric(TestCase, metaclass=Parametrizer):

    testing_data = SasakiMetricTestData()

    def test_inner_product(
        self, metric, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        result = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(result, expected)

    def test_exp(self, metric, tangent_vec, base_point, expected):
        result = metric.exp(tangent_vec, base_point)
        self.assertAllClose(result, expected)

    def test_log(self, metric, point, base_point, expected):
        result = metric.log(point, base_point)
        self.assertAllClose(result, expected)

    def test_geodesic_discrete(self, metric, initial_point, end_point, expected):
        result = metric.geodesic_discrete(initial_point, end_point)
        self.assertAllClose(result, expected, atol=6e-06)
