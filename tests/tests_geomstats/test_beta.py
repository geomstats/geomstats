"""Unit tests for the beta manifold."""


from scipy.stats import beta

import geomstats.backend as gs
import tests.conftest
from tests.conftest import Parametrizer, np_backend, pytorch_backend
from tests.data.beta_data import BetaDistributionsTestsData, BetaMetricTestData
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase

PYTORCH_BACKEND = pytorch_backend()

NOT_AUTOGRAD = pytorch_backend() or np_backend()


class TestBetaDistributions(OpenSetTestCase, metaclass=Parametrizer):
    testing_data = BetaDistributionsTestsData()

    def test_point_to_pdf(self, point, x):
        pdf = self.Space().point_to_pdf(point)
        result = pdf(x)
        expected = gs.transpose(
            gs.array([beta.pdf(x_, a=point[..., 0], b=point[..., 1]) for x_ in x])
        )
        self.assertAllClose(result, expected)


class TestBetaMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_exp_shape = True  # because several base points for one vector
    skip_test_log_shape = PYTORCH_BACKEND
    skip_test_exp_belongs = PYTORCH_BACKEND
    skip_test_log_is_tangent = PYTORCH_BACKEND
    skip_test_dist_is_symmetric = PYTORCH_BACKEND
    skip_test_dist_is_positive = PYTORCH_BACKEND
    skip_test_squared_dist_is_symmetric = True
    skip_test_squared_dist_is_positive = PYTORCH_BACKEND
    skip_test_dist_is_norm_of_log = PYTORCH_BACKEND
    skip_test_dist_point_to_itself_is_zero = PYTORCH_BACKEND
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
    skip_test_estimate_and_belongs_se = True

    testing_data = BetaMetricTestData()
    Space = testing_data.Space

    def test_metric_matrix(self, space, point, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.metric_matrix(point)
        self.assertAllClose(result, expected)

    @tests.conftest.np_only
    def test_exp(self, space, n_samples):
        """Test Exp.

        Test that the Riemannian exponential at points on the first
        bisector computed in the direction of the first bisector stays
        on the first bisector.
        """
        space.equip_with_metric(self.Metric)
        points = space.random_point(n_samples)
        vectors = space.random_point(n_samples)
        initial_vectors = gs.array([[vec_x, vec_x] for vec_x in vectors[:, 0]])
        points = gs.array([[param_a, param_a] for param_a in points[:, 0]])
        result_points = space.metric.exp(initial_vectors, points)
        result = gs.isclose(result_points[:, 0], result_points[:, 1])
        expected = gs.array([True] * n_samples)
        self.assertAllClose(expected, result)

    def test_christoffels_shape(self, space, n_samples):
        """Test Christoffel synbols.
        Check vectorization of Christoffel symbols.
        """
        space.equip_with_metric(self.Metric)
        points = space.random_point(n_samples)
        dim = space.dim
        christoffel = space.metric.christoffels(points)
        result = christoffel.shape
        expected = gs.array([n_samples, dim, dim, dim])
        self.assertAllClose(result, expected)

    @tests.conftest.autograd_only
    def test_sectional_curvature(self, space, n_samples, atol):
        space.equip_with_metric(self.Metric)
        point = space.random_point(n_samples)
        tangent_vec_a = space.metric.random_unit_tangent_vec(point)
        tangent_vec_b = space.metric.random_unit_tangent_vec(point)
        x, y = point[:, 0], point[:, 1]
        detg = gs.polygamma(1, x) * gs.polygamma(1, y) - gs.polygamma(1, x + y) * (
            gs.polygamma(1, x) + gs.polygamma(1, y)
        )
        expected = (
            gs.polygamma(2, x)
            * gs.polygamma(2, y)
            * gs.polygamma(2, x + y)
            * (
                gs.polygamma(1, x) / gs.polygamma(2, x)
                + gs.polygamma(1, y) / gs.polygamma(2, y)
                - gs.polygamma(1, x + y) / gs.polygamma(2, x + y)
            )
            / (4 * detg**2)
        )
        result = self.Space().metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, point
        )
        self.assertAllClose(result, expected, atol)
