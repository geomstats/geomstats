"""Unit tests for the Gamma manifold."""

from scipy.stats import gamma

import geomstats.backend as gs
import geomstats.tests
from geomstats.information_geometry.gamma import GammaDistributions, GammaMetric
from tests.conftest import Parametrizer
from tests.data.gamma_data import GammaMetricTestData, GammaTestData
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestGamma(OpenSetTestCase, metaclass=Parametrizer):

    space = GammaDistributions
    testing_data = GammaTestData()

    def test_belongs(self, vec, expected):
        self.assertAllClose(self.space().belongs(gs.array(vec)), expected)

    def test_random_point(self, point, expected):
        self.assertAllClose(point.shape, expected)

    def test_sample(self, point, n_samples):
        expected = (
            (n_samples,) if len(point.shape) == 1 else (point.shape[0], n_samples)
        )
        self.assertAllClose(self.space().sample(point, n_samples).shape, expected)

    @geomstats.tests.np_and_autograd_only
    def test_point_to_pdf(self, point, n_samples):
        point = gs.to_ndarray(point, 2)
        n_points = point.shape[0]
        pdf = self.space().point_to_pdf(point)
        alpha = gs.ones(2)
        samples = self.space().sample(alpha, n_samples)
        result = pdf(samples)
        pdf = []
        for i in range(n_points):
            pdf.append(
                gs.array(
                    [
                        gamma.pdf(x, a=point[i, 0], scale=point[i, 1] / point[i, 0])
                        for x in samples
                    ]
                )
            )
        expected = gs.squeeze(gs.stack(pdf, axis=0))
        self.assertAllClose(result, expected)

    def test_maximum_likelihood_fit(self, sample, expected):
        result = self.space().maximum_likelihood_fit(sample)
        self.assertAllClose(result, expected)

    def test_natural_to_standard(self, point, expected):
        result = self.space().natural_to_standard(point)
        self.assertAllClose(result, expected)

    def test_natural_to_standard_vectorization(self, point):
        result = self.space().natural_to_standard(point).shape
        expected = point.shape
        self.assertAllClose(result, expected)

    def test_standard_to_natural(self, point, expected):
        result = self.space().standard_to_natural(point)
        self.assertAllClose(result, expected)

    def test_standard_to_natural_vectorization(self, point):
        result = self.space().standard_to_natural(point).shape
        expected = point.shape
        self.assertAllClose(result, expected)

    def test_tangent_natural_to_standard(self, vec, point, expected):
        result = self.space().tangent_natural_to_standard(vec, point)
        self.assertAllClose(result, expected)

    def test_tangent_natural_to_standard_vectorization(self, vec, point):
        result = self.space().tangent_natural_to_standard(vec, point).shape
        expected = vec.shape
        self.assertAllClose(result, expected)

    def test_tangent_standard_to_natural(self, vec, point, expected):
        result = self.space().tangent_standard_to_natural(vec, point)
        self.assertAllClose(result, expected)

    def test_tangent_standard_to_natural_vectorization(self, vec, point):
        result = self.space().tangent_standard_to_natural(vec, point).shape
        expected = vec.shape
        self.assertAllClose(result, expected)


class TestGammaMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    space = GammaDistributions
    connection = metric = GammaMetric

    skip_test_exp_shape = True
    skip_test_log_shape = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_exp_belongs = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_log_is_tangent = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_dist_is_symmetric = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_dist_is_positive = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_squared_dist_is_symmetric = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_squared_dist_is_positive = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_dist_is_norm_of_log = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_dist_point_to_itself_is_zero = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )
    skip_test_exp_after_log = True
    skip_test_log_after_exp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_triangle_inequality_of_dist = (
        geomstats.tests.tf_backend() or geomstats.tests.pytorch_backend()
    )

    testing_data = GammaMetricTestData()

    @geomstats.tests.np_autograd_and_torch_only
    def test_metric_matrix_shape(self, point, expected):
        return self.assertAllClose(self.metric().metric_matrix(point).shape, expected)

    @geomstats.tests.np_autograd_and_tf_only
    def test_christoffels_vectorization(self, point, expected):
        return self.assertAllClose(self.metric().christoffels(point), expected)

    @geomstats.tests.np_autograd_and_tf_only
    def test_christoffels_shape(self, point, expected):
        return self.assertAllClose(
            self.metric().christoffels(base_point=point).shape,
            expected,
        )

    @geomstats.tests.np_and_autograd_only
    def test_exp_vectorization(self, point, tangent_vecs, exp_solver):
        """Test the case with one initial point and several tangent vectors."""
        end_points = self.metric().exp(
            tangent_vec=tangent_vecs, base_point=point, n_steps=1000, solver=exp_solver
        )
        result = end_points.shape
        expected = (tangent_vecs.shape[0], 2)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_autograd_only
    def test_exp_control(self, base_point, tangent_vec, exp_solver):
        """Test exp at a random base point for a tangent vector of controlled norm."""
        end_point = self.metric().exp(
            tangent_vec=tangent_vec,
            base_point=base_point,
            n_steps=1000,
            solver=exp_solver,
        )
        result = gs.any(gs.isnan(end_point))
        self.assertAllClose(result, False)

    @geomstats.tests.autograd_only
    def test_log_control(self, base_point, tangent_vec, exp_solver, log_method):
        """Test log at a pair of points with controlled geodesic distance."""
        point = self.metric().exp(
            self.metric().normalize(vector=tangent_vec, base_point=base_point),
            base_point,
            solver=exp_solver,
        )
        vec = self.metric().log(point, base_point, n_steps=1000, method=log_method)
        result = gs.any(gs.isnan(vec))
        self.assertAllClose(result, False)

    @geomstats.tests.autograd_only
    def test_exp_after_log_control(
        self, base_point, end_point, exp_solver, log_method, rtol
    ):
        """Test exp after log at a pair of points with controlled geodesic distance."""
        expected = end_point
        tangent_vec = self.metric().log(
            expected, base_point, n_steps=1000, method=log_method
        )
        end_point = self.metric().exp(
            tangent_vec, base_point, n_steps=1000, solver=exp_solver
        )
        result = end_point
        self.assertAllClose(result, expected, rtol=rtol)

    @geomstats.tests.autograd_only
    def test_log_after_exp_control(
        self, base_point, tangent_vec, exp_solver, log_method, rtol
    ):
        """Test exp after log at a pair of points with controlled geodesic distance."""
        expected = tangent_vec
        end_point = self.metric().exp(
            expected, base_point, n_steps=1000, solver=exp_solver
        )
        back_to_vec = self.metric().log(
            end_point, base_point, n_steps=1000, method=log_method
        )
        result = back_to_vec
        self.assertAllClose(result, expected, rtol=rtol)

    @geomstats.tests.autograd_and_torch_only
    def test_jacobian_christoffels(self, point):
        n_points = 1 if len(point.shape) == 1 else point.shape[0]

        result = self.metric().jacobian_christoffels(point[0, :])
        self.assertAllClose((2, 2, 2, 2), result.shape)

        expected = gs.autodiff.jacobian(self.metric().christoffels)(point[0, :])
        self.assertAllClose(expected, result)

        result = self.metric().jacobian_christoffels(point)
        expected = [
            self.metric().jacobian_christoffels(point[i, :]) for i in range(n_points)
        ]
        expected = gs.stack(expected, 0)
        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_autograd_only
    def test_geodesic(self, base_point, norm, solver):
        """Check that the norm of the geodesic velocity is constant."""
        n_steps = 1000
        tangent_vec = norm * self.metric().random_unit_tangent_vec(
            base_point=base_point
        )
        geod = self.metric().geodesic(
            initial_point=base_point,
            initial_tangent_vec=tangent_vec,
            n_steps=1000,
            solver=solver,
        )
        t = gs.linspace(0.0, 1.0, n_steps)
        geod_at_t = geod(t)
        velocity = n_steps * (geod_at_t[1:, :] - geod_at_t[:-1, :])
        velocity_norm = self.metric().norm(velocity, geod_at_t[:-1, :])
        result = 1 / velocity_norm.min() * (velocity_norm.max() - velocity_norm.min())
        expected = 0.0
        self.assertAllClose(expected, result, rtol=1.0)

    @geomstats.tests.np_and_autograd_only
    def test_geodesic_shape(self, point, n_vec, norm, time, solver, expected):
        tangent_vec = norm * self.metric().random_unit_tangent_vec(
            base_point=point, n_vectors=n_vec
        )
        geod = self.metric().geodesic(
            initial_point=point,
            initial_tangent_vec=tangent_vec,
            n_steps=1000,
            solver=solver,
        )
        result = geod(time).shape
        self.assertAllClose(expected, result)

    @geomstats.tests.autograd_only
    def test_scalar_curvature(self, point, atol):
        kappa = point[..., 0]
        expected = (gs.polygamma(1, kappa) + kappa * gs.polygamma(2, kappa)) / (
            2 * (-1 + kappa * gs.polygamma(1, kappa)) ** 2
        )
        result = self.metric().scalar_curvature(point)
        self.assertAllClose(expected, result, atol)
