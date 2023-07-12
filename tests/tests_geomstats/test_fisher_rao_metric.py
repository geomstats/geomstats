"""Unit tests for the fisher rao metric."""


import geomstats.backend as gs
import tests.conftest
from tests.conftest import Parametrizer
from tests.data.fisher_rao_metric_data import FisherRaoMetricTestData
from tests.geometry_test_cases import TestCase


@tests.conftest.autograd_and_torch_only
class TestFisherRaoMetric(TestCase, metaclass=Parametrizer):
    testing_data = FisherRaoMetricTestData()

    Metric = testing_data.Metric

    def test_inner_product_matrix_shape(self, space, support, base_point):
        space.equip_with_metric(self.Metric, support=support)
        dim = space.dim
        result = space.metric.metric_matrix(base_point=base_point)
        if base_point.ndim == 1:
            self.assertAllClose(gs.shape(result), (dim, dim))
        else:
            self.assertAllClose(gs.shape(result), (base_point.shape[0], dim, dim))

    def test_det_of_inner_product_matrix(self, space, support, base_point):
        space.equip_with_metric(self.Metric, support=support)
        inner_prod_mat = space.metric.metric_matrix(base_point=base_point)
        result = gs.linalg.det(inner_prod_mat)
        if base_point.ndim == 1:
            self.assertTrue(result > 0.0)
        else:
            for result_ in result:
                self.assertTrue(result_ > 0.0)

    def test_metric_matrix_and_closed_form_metric_matrix(
        self,
        space,
        support,
        base_point,
    ):
        space.equip_with_metric(self.Metric, support=support)
        closed_form_metric = space.default_metric()(space)

        inner_prod_mat = space.metric.metric_matrix(
            base_point=base_point,
        )
        normal_metric_mat = closed_form_metric.metric_matrix(
            base_point=base_point,
        )
        self.assertAllClose(inner_prod_mat, normal_metric_mat)

    def test_inner_product_and_closed_form_inner_product(
        self,
        space,
        support,
        tangent_vec_a,
        tangent_vec_b,
        base_point,
    ):
        space.equip_with_metric(self.Metric, support=support)
        closed_form_metric = space.default_metric()(space)

        inner_prod_mat = space.metric.inner_product(
            tangent_vec_a=tangent_vec_a,
            tangent_vec_b=tangent_vec_b,
            base_point=base_point,
        )
        normal_metric_mat = closed_form_metric.inner_product(
            tangent_vec_a=tangent_vec_a,
            tangent_vec_b=tangent_vec_b,
            base_point=base_point,
        )
        self.assertAllClose(inner_prod_mat, normal_metric_mat)

    def test_inner_product_derivative_and_closed_form_inner_product_derivative(
        self,
        space,
        support,
        closed_form_derivative,
        base_point,
    ):
        space.equip_with_metric(self.Metric, support=support)

        inner_prod_deriv_mat = space.metric.inner_product_derivative_matrix(
            base_point=base_point
        )
        normal_inner_prod_deriv_mat = closed_form_derivative(base_point)
        self.assertAllClose(inner_prod_deriv_mat, normal_inner_prod_deriv_mat)
