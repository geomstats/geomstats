"""Unit tests for the categorical manifold."""

import geomstats.backend as gs
from tests.conftest import Parametrizer, np_backend, pytorch_backend, tf_backend
from tests.data.multinomial_data import MultinomialMetricTestData, MultinomialTestData
from tests.geometry_test_cases import LevelSetTestCase, RiemannianMetricTestCase

TF_OR_PYTORCH_BACKEND = tf_backend() or pytorch_backend()

NOT_AUTOGRAD = tf_backend() or pytorch_backend() or np_backend()


class TestMultinomialDistributions(LevelSetTestCase, metaclass=Parametrizer):
    """Class defining the categorical distributions tests."""

    skip_test_extrinsic_after_intrinsic = True
    skip_test_intrinsic_after_extrinsic = True
    testing_data = MultinomialTestData()

    def test_sample_shape(self, dim, n_draws, point, n_samples, expected):
        self.assertAllClose(
            self.Space(dim, n_draws).sample(point, n_samples).shape, expected
        )


class TestMultinomialMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_log_after_exp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_ivp_belongs = tf_backend()
    skip_test_geodesic_bvp_belongs = tf_backend()
    skip_test_exp_geodesic_ivp = NOT_AUTOGRAD
    skip_test_exp_ladder_parallel_transport = True
    skip_test_riemann_tensor_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_sectional_curvature_shape = tf_backend()
    skip_test_sectional_curvature_is_positive = tf_backend()

    testing_data = MultinomialMetricTestData()
    Space = testing_data.Space

    def test_sectional_curvature_is_positive(self, dim, n_draws, base_point):
        space = self.Space(dim, n_draws)
        metric = self.Metric(dim, n_draws)
        tangent_vec = space.to_tangent(gs.random.rand(dim + 1), base_point)
        result = metric.sectional_curvature(tangent_vec, tangent_vec, base_point)
        self.assertAllClose(gs.all(result > 0), True)
