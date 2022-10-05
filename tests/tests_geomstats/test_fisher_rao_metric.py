"""Unit tests for the fisher rao metric."""


import geomstats.backend as gs
import tests.conftest
from tests.conftest import Parametrizer, autograd_backend, np_backend
from tests.data.fisher_rao_metric_data import FisherRaoMetricTestData
from tests.geometry_test_cases import TestCase


@tests.conftest.autograd_tf_and_torch_only
class TestFisherRaoMetric(TestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_shape = np_backend()
    skip_test_geodesic_ivp_belongs = True
    skip_test_exp_ladder_parallel_transport = np_backend()
    skip_test_log_is_tangent = np_backend()
    skip_test_log_shape = np_backend()
    skip_test_geodesic_bvp_belongs = np_backend()
    skip_test_exp_after_log = np_backend() or autograd_backend()
    skip_test_geodesic_bvp_belongs = True
    skip_test_log_after_exp = True
    skip_test_dist_point_to_itself_is_zero = np_backend()
    skip_test_triangle_inequality_of_dist = True
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
        print("MAT")
        print(inner_prod_mat)
        inv_inner_prod_mat = gs.linalg.inv(inner_prod_mat)
        result = gs.matmul(inv_inner_prod_mat, inner_prod_mat)
        expected = gs.eye(information_manifold.dim)
        self.assertAllClose(result, expected)
