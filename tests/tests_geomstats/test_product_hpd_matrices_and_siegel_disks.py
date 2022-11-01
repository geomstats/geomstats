"""Unit tests for the ProductHPDMatricesAndSiegelDisks manifold."""

from tests.conftest import Parametrizer, TestCase
from tests.data.product_hpd_matrices_and_siegel_disks_data import (
    ProductHPDMatricesAndSiegelDisksMetricTestData,
    ProductHPDMatricesAndSiegelDisksTestData,
)
from tests.geometry_test_cases import OpenSetTestCase


class TestProductHPDMatricesAndSiegelDisks(OpenSetTestCase, metaclass=Parametrizer):

    skip_test_to_tangent_is_tangent_in_embedding_space = True
    skip_test_to_tangent_is_tangent = True

    testing_data = ProductHPDMatricesAndSiegelDisksTestData()

    def test_dimension(self, n_manifolds, n, expected):
        space = self.Space(n_manifolds, n)
        self.assertAllClose(space.dim, expected)


class TestProductHPDMatricesAndSiegelDisksMetric(TestCase, metaclass=Parametrizer):

    testing_data = ProductHPDMatricesAndSiegelDisksMetricTestData()
    Metric = testing_data.Metric

    def test_signature(self, n_manifolds, n, expected):
        metric = self.Metric(n_manifolds, n)
        self.assertAllClose(metric.signature, expected)
