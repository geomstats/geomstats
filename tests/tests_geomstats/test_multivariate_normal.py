"""Unit tests for the MultivariateDiagonalNormalDistributions manifold."""

from tests.conftest import Parametrizer, np_backend, pytorch_backend, tf_backend
from tests.data.multivariate_normal import (
    MultivariateDiagonalNormalDistributionsTestData,
    MultivariateDiagonalNormalMetricTestData,
)
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase

TF_OR_PYTORCH_BACKEND = tf_backend() or pytorch_backend()

NOT_AUTOGRAD = tf_backend() or pytorch_backend() or np_backend()


class TestMultivariateDiagonalNormalDistributions(
    OpenSetTestCase, metaclass=Parametrizer
):
    testing_data = MultivariateDiagonalNormalDistributionsTestData()

    def test_belongs(self, n, point, expected):
        self.assertAllClose(self.Space(n).belongs(point), expected)

    def test_random_point_shape(self, point, expected):
        self.assertAllClose(point.shape, expected)

    def test_sample(self, n, point, n_samples, expected):
        self.assertAllClose(self.Space(n).sample(point, n_samples).shape, expected)


class TestMultivariateDiagonalNormalMetric(
    RiemannianMetricTestCase, metaclass=Parametrizer
):
    skip_test_exp_shape = True  # because several base points for one vector
    skip_test_log_shape = TF_OR_PYTORCH_BACKEND
    # skip_test_exp_belongs = TF_OR_PYTORCH_BACKEND
    skip_test_exp_belongs = True
    skip_test_log_is_tangent = TF_OR_PYTORCH_BACKEND
    skip_test_dist_is_symmetric = TF_OR_PYTORCH_BACKEND
    skip_test_dist_is_positive = TF_OR_PYTORCH_BACKEND
    skip_test_squared_dist_is_symmetric = True
    skip_test_squared_dist_is_positive = TF_OR_PYTORCH_BACKEND
    skip_test_dist_is_norm_of_log = TF_OR_PYTORCH_BACKEND
    skip_test_dist_point_to_itself_is_zero = TF_OR_PYTORCH_BACKEND
    skip_test_log_after_exp = True
    skip_test_exp_after_log = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_triangle_inequality_of_dist = True
    skip_test_riemann_tensor_shape = NOT_AUTOGRAD
    skip_test_ricci_tensor_shape = NOT_AUTOGRAD
    skip_test_scalar_curvature_shape = NOT_AUTOGRAD
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = NOT_AUTOGRAD
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = NOT_AUTOGRAD
    skip_test_covariant_riemann_tensor_bianchi_identity = NOT_AUTOGRAD
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = NOT_AUTOGRAD
    skip_test_sectional_curvature_shape = NOT_AUTOGRAD

    testing_data = MultivariateDiagonalNormalMetricTestData()
    Space = testing_data.Space
