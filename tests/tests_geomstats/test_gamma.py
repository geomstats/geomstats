"""Unit tests for the Gamma manifold."""

import pytest
from scipy.stats import gamma

import geomstats.backend as gs
import tests.conftest
from tests.conftest import Parametrizer, np_backend
from tests.data.gamma_data import GammaDistributionsTestData, GammaMetricTestData
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase

NOT_AUTODIFF = np_backend()


class TestGammaDistributions(OpenSetTestCase, metaclass=Parametrizer):
    testing_data = GammaDistributionsTestData()

    def test_belongs(self, vec, expected):
        self.assertAllClose(self.Space().belongs(vec), expected)

    def test_random_point(self, point, expected):
        self.assertAllClose(point.shape, expected)

    def test_sample(self, point, n_samples):
        expected = (
            (n_samples,) if len(point.shape) == 1 else (point.shape[0], n_samples)
        )
        self.assertAllClose(self.Space().sample(point, n_samples).shape, expected)

    def test_point_to_pdf(self, point, n_samples):
        pdf = self.Space().point_to_pdf(point)
        result = pdf(n_samples)
        expected = gs.transpose(
            gs.array(
                [
                    gamma.pdf(x_, a=point[..., 0], scale=point[..., 1] / point[..., 0])
                    for x_ in n_samples
                ]
            )
        )
        self.assertAllClose(result, expected)

    def test_maximum_likelihood_fit(self, sample, expected):
        result = self.Space().maximum_likelihood_fit(sample)
        self.assertAllClose(result, expected)

    def test_natural_to_standard(self, point, expected):
        result = self.Space().natural_to_standard(point)
        self.assertAllClose(result, expected)

    def test_natural_to_standard_vectorization(self, point):
        result = self.Space().natural_to_standard(point).shape
        expected = point.shape
        self.assertAllClose(result, expected)

    def test_standard_to_natural(self, point, expected):
        result = self.Space().standard_to_natural(point)
        self.assertAllClose(result, expected)

    def test_standard_to_natural_vectorization(self, point):
        result = self.Space().standard_to_natural(point).shape
        expected = point.shape
        self.assertAllClose(result, expected)

    def test_tangent_natural_to_standard(self, vec, point, expected):
        result = self.Space().tangent_natural_to_standard(vec, point)
        self.assertAllClose(result, expected)

    def test_tangent_natural_to_standard_vectorization(self, vec, point):
        result = self.Space().tangent_natural_to_standard(vec, point).shape
        expected = vec.shape
        self.assertAllClose(result, expected)

    def test_tangent_standard_to_natural(self, vec, point, expected):
        result = self.Space().tangent_standard_to_natural(vec, point)
        self.assertAllClose(result, expected)

    def test_tangent_standard_to_natural_vectorization(self, vec, point):
        result = self.Space().tangent_standard_to_natural(vec, point).shape
        expected = vec.shape
        self.assertAllClose(result, expected)


class TestGammaMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_exp_shape = True  # because several base points for one vector
    skip_test_exp_belongs = True
    skip_test_dist_is_symmetric = True
    skip_test_squared_dist_is_symmetric = True
    skip_test_dist_is_norm_of_log = True
    skip_test_dist_point_to_itself_is_zero = True
    skip_test_log_after_exp = True
    skip_test_exp_after_log = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_triangle_inequality_of_dist = True
    skip_test_riemann_tensor_shape = NOT_AUTODIFF
    skip_test_ricci_tensor_shape = NOT_AUTODIFF
    skip_test_scalar_curvature_shape = NOT_AUTODIFF
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = NOT_AUTODIFF
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = NOT_AUTODIFF
    skip_test_covariant_riemann_tensor_bianchi_identity = NOT_AUTODIFF
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = NOT_AUTODIFF
    skip_test_sectional_curvature_shape = NOT_AUTODIFF

    testing_data = GammaMetricTestData()

    @pytest.mark.xfail
    def test_covariant_riemann_tensor_is_interchange_symmetric(
        self, space, metric_args, base_point
    ):
        return super().test_covariant_riemann_tensor_is_interchange_symmetric(
            space, metric_args, base_point
        )

    @pytest.mark.xfail
    def test_covariant_riemann_tensor_is_skew_symmetric_2(
        self, space, metric_args, base_point
    ):
        return super().test_covariant_riemann_tensor_is_skew_symmetric_2(
            space, metric_args, base_point
        )

    @pytest.mark.xfail
    def test_covariant_riemann_tensor_bianchi_identity(
        self, space, metric_args, base_point
    ):
        return super().test_covariant_riemann_tensor_bianchi_identity(
            space, metric_args, base_point
        )

    def test_metric_matrix_shape(self, space, n_points, expected):
        space.equip_with_metric(self.Metric)
        point = space.random_point(n_points)
        return self.assertAllClose(space.metric.metric_matrix(point).shape, expected)

    def test_christoffels_vectorization(self, space, n_points):
        space.equip_with_metric(self.Metric)

        points = space.random_point(n_points)
        christoffel_1 = space.metric.christoffels(base_point=points[0])
        christoffel_2 = space.metric.christoffels(base_point=points[1])
        expected = gs.stack((christoffel_1, christoffel_2), axis=0)

        return self.assertAllClose(space.metric.christoffels(points), expected)

    def test_christoffels_shape(self, space, n_points, expected):
        space.equip_with_metric(self.Metric)
        point = space.random_point(n_points)
        return self.assertAllClose(
            space.metric.christoffels(base_point=point).shape,
            expected,
        )

    def test_exp_vectorization(self, space, point, tangent_vecs):
        """Test the case with one initial point and several tangent vectors."""
        space.equip_with_metric(self.Metric)
        end_points = space.metric.exp(
            tangent_vec=tangent_vecs,
            base_point=point,
        )
        result = end_points.shape
        expected = (tangent_vecs.shape[0], 2)
        self.assertAllClose(result, expected)

    def test_exp_control(self, space, base_point, tangent_vec):
        """Test exp at a random base point for a tangent vector of controlled norm."""
        space.equip_with_metric(self.Metric)
        end_point = space.metric.exp(
            tangent_vec=tangent_vec,
            base_point=base_point,
        )
        result = gs.any(gs.isnan(end_point))
        self.assertAllClose(result, False)

    def test_log_control(self, space, base_point, tangent_vec):
        """Test log at a pair of points with controlled geodesic distance."""
        space.equip_with_metric(self.Metric)
        point = space.metric.exp(
            space.metric.normalize(vector=tangent_vec, base_point=base_point),
            base_point,
        )
        vec = space.metric.log(point, base_point)
        result = gs.any(gs.isnan(vec))
        self.assertAllClose(result, False)

    @pytest.mark.xfail
    def test_exp_after_log_control(self, space, base_point, end_point, atol):
        """Test exp after log at a pair of points with controlled geodesic distance."""
        space.equip_with_metric(self.Metric)
        expected = end_point
        tangent_vec = space.metric.log(expected, base_point)
        end_point = space.metric.exp(tangent_vec, base_point)
        result = end_point
        self.assertAllClose(result, expected, atol=atol)

    @pytest.mark.xfail
    def test_log_after_exp_control(self, space, base_point, tangent_vec, atol):
        """Test exp after log at a pair of points with controlled geodesic distance."""
        space.equip_with_metric(self.Metric)
        expected = tangent_vec
        end_point = space.metric.exp(expected, base_point)
        back_to_vec = space.metric.log(end_point, base_point)
        result = back_to_vec
        self.assertAllClose(result, expected, atol=atol)

    @tests.conftest.autograd_and_torch_only
    def test_jacobian_christoffels(self, space, n_points):
        space.equip_with_metric(self.Metric)

        point = space.random_point(n_points)

        result = space.metric.jacobian_christoffels(point[0, :])
        self.assertAllClose((2, 2, 2, 2), result.shape)

        expected = gs.autodiff.jacobian(space.metric.christoffels)(point[0, :])
        self.assertAllClose(expected, result)

        result = space.metric.jacobian_christoffels(point)
        expected = [
            space.metric.jacobian_christoffels(point[i, :]) for i in range(n_points)
        ]
        expected = gs.stack(expected, 0)
        self.assertAllClose(expected, result)

    def test_geodesic(self, space, norm):
        """Check that the norm of the geodesic velocity is constant."""
        n_steps = 1000
        space.equip_with_metric(self.Metric)
        base_point = space.random_point()
        tangent_vec = norm * space.metric.random_unit_tangent_vec(base_point=base_point)
        geod = space.metric.geodesic(
            initial_point=base_point,
            initial_tangent_vec=tangent_vec,
        )
        t = gs.linspace(0.0, 1.0, n_steps)
        geod_at_t = geod(t)
        velocity = n_steps * (geod_at_t[1:, :] - geod_at_t[:-1, :])
        velocity_norm = space.metric.norm(velocity, geod_at_t[:-1, :])
        result = 1 / velocity_norm.min() * (velocity_norm.max() - velocity_norm.min())
        expected = 0.0
        self.assertAllClose(expected, result, rtol=1.0)

    def test_geodesic_shape(self, space, n_vec, norm, time, expected):
        space.equip_with_metric(self.Metric)
        point = space.random_point()
        tangent_vec = norm * space.metric.random_unit_tangent_vec(
            base_point=point, n_vectors=n_vec
        )
        geod = space.metric.geodesic(
            initial_point=point,
            initial_tangent_vec=tangent_vec,
        )
        result = geod(time).shape
        self.assertAllClose(expected, result)

    @tests.conftest.autograd_and_torch_only
    def test_scalar_curvature(self, space, n_points, atol):
        space.equip_with_metric(self.Metric)
        point = space.random_point(n_points)
        kappa = point[..., 0]
        expected = (gs.polygamma(1, kappa) + kappa * gs.polygamma(2, kappa)) / (
            2 * (-1 + kappa * gs.polygamma(1, kappa)) ** 2
        )
        result = space.metric.scalar_curvature(point)
        self.assertAllClose(expected, result, atol)
