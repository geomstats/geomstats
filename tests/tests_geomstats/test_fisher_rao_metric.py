"""Unit tests for the fisher rao metric."""


import geomstats.backend as gs
import tests.conftest
from tests.conftest import Parametrizer
from tests.data.fisher_rao_metric_data import FisherRaoMetricTestData
from tests.geometry_test_cases import TestCase


@tests.conftest.autograd_and_torch_only
# Note: it also works in tensorflow but it is insanely slow.
class TestFisherRaoMetric(TestCase, metaclass=Parametrizer):
    testing_data = FisherRaoMetricTestData()

    Metric = testing_data.Metric

    def test_inner_product_matrix_shape(
        self, information_manifold, support, base_point
    ):
        metric = self.Metric(information_manifold=information_manifold, support=support)
        dim = metric.dim
        result = metric.metric_matrix(base_point=base_point)
        self.assertAllClose(gs.shape(result), (dim, dim))

    def test_inner_product_matrix_and_its_inverse(
        self, information_manifold, support, base_point
    ):
        metric = self.Metric(information_manifold=information_manifold, support=support)
        inner_prod_mat = metric.metric_matrix(base_point=base_point)
        inv_inner_prod_mat = gs.linalg.inv(inner_prod_mat)
        result = gs.matmul(inv_inner_prod_mat, inner_prod_mat)
        expected = gs.eye(information_manifold.dim)
        self.assertAllClose(result, expected)

    def test_metric_matrix_and_closed_form_metric_matrix(
        self,
        information_manifold,
        support,
        closed_form_metric,
        base_point,
    ):
        metric = self.Metric(information_manifold=information_manifold, support=support)
        inner_prod_mat = metric.metric_matrix(
            base_point=base_point,
        )
        normal_metric_mat = closed_form_metric.metric_matrix(
            base_point=base_point,
        )
        self.assertAllClose(inner_prod_mat, normal_metric_mat)
