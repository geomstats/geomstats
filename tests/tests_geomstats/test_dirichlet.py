"""Unit tests for the Dirichlet manifold."""

import pytest
from scipy.stats import dirichlet

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.conftest import Parametrizer, np_backend
from tests.data.dirichlet_data import DirichletMetricTestData, DirichletTestData
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase

NOT_AUTODIFF = np_backend()


class TestDirichlet(OpenSetTestCase, metaclass=Parametrizer):
    testing_data = DirichletTestData()

    def test_belongs(self, dim, vec, expected):
        self.assertAllClose(self.Space(dim).belongs(vec), expected)

    def test_random_point(self, point, expected):
        self.assertAllClose(point.shape, expected)

    def test_sample(self, dim, point, n_samples, expected):
        self.assertAllClose(self.Space(dim).sample(point, n_samples).shape, expected)

    def test_sample_belongs(self, dim, point, n_samples, expected):
        samples = self.Space(dim).sample(point, n_samples)
        self.assertAllClose(gs.sum(samples, axis=-1), expected)

    def test_point_to_pdf(self, dim, point, n_samples):
        space = self.Space(dim)
        point = gs.to_ndarray(point, 2)
        n_points = point.shape[0]
        pdf = space.point_to_pdf(point)
        alpha = gs.ones(dim)
        samples = space.sample(alpha, n_samples)
        result = pdf(samples)
        pdf = []
        for i in range(n_points):
            pdf.append(gs.array([dirichlet.pdf(x, point[i, :]) for x in samples]))
        expected = gs.stack(pdf, axis=0)
        self.assertAllClose(result, expected)


class TestDirichletMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_exp_shape = True  # because several base points for one vector
    skip_test_squared_dist_is_symmetric = True
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

    testing_data = DirichletMetricTestData()
    Metric = testing_data.Metric

    @pytest.mark.xfail
    def test_exp_belongs(self, connection_args, space, tangent_vec, base_point, atol):
        return super().test_exp_belongs(
            connection_args, space, tangent_vec, base_point, atol
        )

    @pytest.mark.xfail
    def test_dist_is_symmetric(self, space, metric_args, point_a, point_b, rtol, atol):
        return super().test_dist_is_symmetric(
            space, metric_args, point_a, point_b, rtol, atol
        )

    @pytest.mark.xfail
    def test_dist_is_norm_of_log(
        self, space, metric_args, point_a, point_b, rtol, atol
    ):
        return super().test_dist_is_norm_of_log(
            space, metric_args, point_a, point_b, rtol, atol
        )

    @pytest.mark.xfail
    def test_covariant_riemann_tensor_is_interchange_symmetric(
        self, space, metric_args, base_point
    ):
        return super().test_covariant_riemann_tensor_is_interchange_symmetric(
            space, metric_args, base_point
        )

    @pytest.mark.xfail
    def test_covariant_riemann_tensor_is_skew_symmetric_1(
        self, space, metric_args, base_point
    ):
        return super().test_covariant_riemann_tensor_is_skew_symmetric_2(
            space, metric_args, base_point
        )

    @pytest.mark.xfail
    def test_covariant_riemann_tensor_is_skew_symmetric_2(
        self, space, metric_args, base_point
    ):
        return super().test_covariant_riemann_tensor_is_skew_symmetric_2(
            space, metric_args, base_point
        )

    def test_metric_matrix_shape(self, space, n_points, expected):
        space.equip_with_metric(self.Metric)
        point = space.random_point(n_points)
        return self.assertAllClose(space.metric.metric_matrix(point).shape, expected)

    def test_metric_matrix_dim_2(self, space, n_points):
        space.equip_with_metric(self.Metric)
        point = space.random_point(n_points)

        param_a = point[..., 0]
        param_b = point[..., 1]
        vector = gs.stack(
            [
                gs.polygamma(1, param_a) - gs.polygamma(1, param_a + param_b),
                -gs.polygamma(1, param_a + param_b),
                gs.polygamma(1, param_b) - gs.polygamma(1, param_a + param_b),
            ],
            axis=-1,
        )
        expected = SymmetricMatrices.from_vector(vector)
        return self.assertAllClose(space.metric.metric_matrix(point), expected)

    def test_christoffels_vectorization(self, space):
        space.equip_with_metric(self.Metric)
        n_points = 2
        points = space.random_point(n_points)
        christoffel_1 = space.metric.christoffels(points[0, :])
        christoffel_2 = space.metric.christoffels(points[1, :])
        expected = gs.stack((christoffel_1, christoffel_2), axis=0)

        return self.assertAllClose(space.metric.christoffels(points), expected)

    def test_christoffels_shape(self, space, n_points, expected):
        space.equip_with_metric(self.Metric)
        point = space.random_point(n_points)
        return self.assertAllClose(space.metric.christoffels(point).shape, expected)

    def test_christoffels_dim_2(self, space):
        def coefficients(param_a, param_b):
            """Christoffel coefficients for the beta distributions."""
            poly1a = gs.polygamma(1, param_a)
            poly2a = gs.polygamma(2, param_a)
            poly1b = gs.polygamma(1, param_b)
            poly2b = gs.polygamma(2, param_b)
            poly1ab = gs.polygamma(1, param_a + param_b)
            poly2ab = gs.polygamma(2, param_a + param_b)
            metric_det = 2 * (poly1a * poly1b - poly1ab * (poly1a + poly1b))

            c1 = (poly2a * (poly1b - poly1ab) - poly1b * poly2ab) / metric_det
            c2 = -poly1b * poly2ab / metric_det
            c3 = (poly2b * poly1ab - poly1b * poly2ab) / metric_det
            return c1, c2, c3

        space.equip_with_metric(self.Metric)

        gs.random.seed(123)
        n_points = 3
        points = space.random_point(n_points)
        param_a, param_b = points[:, 0], points[:, 1]
        c1, c2, c3 = coefficients(param_a, param_b)
        c4, c5, c6 = coefficients(param_b, param_a)
        vector_0 = gs.stack([c1, c2, c3], axis=-1)
        vector_1 = gs.stack([c6, c5, c4], axis=-1)
        gamma_0 = SymmetricMatrices.from_vector(vector_0)
        gamma_1 = SymmetricMatrices.from_vector(vector_1)

        expected = gs.stack([gamma_0, gamma_1], axis=-3)
        return self.assertAllClose(space.metric.christoffels(points), expected)

    def test_exp_diagonal(self, space, param, param_list):
        """Check that the diagonal x1 = ... = xn is totally geodesic."""
        space.equip_with_metric(self.Metric)
        base_point = param * gs.ones(space.dim)
        initial_vectors = gs.transpose(gs.tile(param_list, (space.dim, 1)))
        result = space.metric.exp(initial_vectors, base_point)
        expected = gs.squeeze(gs.transpose(gs.tile(result[..., 0], (space.dim, 1))))
        return self.assertAllClose(expected, result)

    @pytest.mark.xfail
    def test_exp_subspace(self, space, vec, point, expected, atol):
        """Check that subspaces xi1 = ... = xik are totally geodesic."""
        space.equip_with_metric(self.Metric)
        end_point = space.metric.exp(vec, point)
        result = gs.isclose(end_point - end_point[0], 0.0, atol=atol)
        return self.assertAllClose(expected, result)

    def test_exp_vectorization(self, space, tangent_vecs):
        """Test the case with one initial point and several tangent vectors."""
        space.equip_with_metric(self.Metric)
        point = space.random_point()

        end_points = space.metric.exp(tangent_vec=tangent_vecs, base_point=point)
        result = end_points.shape
        expected = (tangent_vecs.shape[0], space.dim)
        self.assertAllClose(result, expected)

    def test_exp_after_log(self, space, base_point, point):
        space.equip_with_metric(self.Metric)
        log = space.metric.log(point, base_point, n_steps=500)
        expected = point
        result = space.metric.exp(tangent_vec=log, base_point=base_point)
        self.assertAllClose(result, expected, rtol=1e-2)

    def test_geodesic_ivp_shape(self, space, n_points, n_steps, expected):
        space.equip_with_metric(self.Metric)

        point = space.random_point(n_points)
        vec = space.random_point(n_points)

        t = gs.linspace(0.0, 1.0, n_steps)
        geodesic = space.metric.geodesic(point, initial_tangent_vec=vec)
        geodesic_at_t = geodesic(t)
        result = geodesic_at_t.shape
        return self.assertAllClose(result, expected)

    def test_geodesic_bvp_shape(self, space, n_points, n_steps, expected):
        space.equip_with_metric(self.Metric)

        point_a = space.random_point(n_points)
        point_b = space.random_point(n_points)

        t = gs.linspace(0.0, 1.0, n_steps)
        geodesic = space.metric.geodesic(point_a, end_point=point_b)
        geodesic_at_t = geodesic(t)
        result = geodesic_at_t.shape
        return self.assertAllClose(result, expected)

    def test_geodesic(self, space):
        """Check that the norm of the geodesic velocity is constant."""
        space.equip_with_metric(self.Metric)

        point_a = space.random_point()
        point_b = space.random_point()

        n_steps = 1000
        geod = space.metric.geodesic(initial_point=point_a, end_point=point_b)
        t = gs.linspace(0.0, 1.0, n_steps)
        geod_at_t = geod(t)
        velocity = n_steps * (geod_at_t[1:, :] - geod_at_t[:-1, :])
        velocity_norm = space.metric.norm(velocity, geod_at_t[:-1, :])
        result = 1 / velocity_norm.min() * (velocity_norm.max() - velocity_norm.min())
        expected = 0.0
        return self.assertAllClose(expected, result, rtol=1.0)

    def test_geodesic_shape(self, space, n_points, time, expected):
        space.equip_with_metric(self.Metric)

        point = space.random_point()
        vec = space.random_point(n_points)

        geod = space.metric.geodesic(initial_point=point, initial_tangent_vec=vec)
        result = geod(time).shape
        self.assertAllClose(expected, result)

    @tests.conftest.autograd_and_torch_only
    def test_jacobian_christoffels(self, space, n_points):
        space.equip_with_metric(self.Metric)
        dim = space.dim

        point = space.random_point(n_points)

        result = space.metric.jacobian_christoffels(point[0, :])
        self.assertAllClose((dim, dim, dim, dim), result.shape)

        expected = gs.autodiff.jacobian(space.metric.christoffels)(point[0, :])
        self.assertAllClose(expected, result)

        result = space.metric.jacobian_christoffels(point)
        expected = [
            space.metric.jacobian_christoffels(point[0, :]),
            space.metric.jacobian_christoffels(point[1, :]),
        ]
        expected = gs.stack(expected, 0)
        self.assertAllClose(expected, result)

    @tests.conftest.autograd_and_torch_only
    def test_sectional_curvature_is_negative(self, space):
        space.equip_with_metric(self.Metric)

        base_point = space.random_point()

        tangent_vec_a, tangent_vec_b = space.metric.random_unit_tangent_vec(
            base_point, 2
        )
        result = gs.all(
            space.metric.sectional_curvature(tangent_vec_a, tangent_vec_b, base_point)
            < 0
        )
        self.assertAllClose(result, True)

    @tests.conftest.np_and_autograd_only
    def test_approx_geodesic_bvp(self, space):
        space.equip_with_metric(self.Metric)

        point_a = space.random_point()
        point_b = space.random_point()

        res = space.metric._approx_geodesic_bvp(point_a, point_b)
        result = res[0]
        expected = space.metric.dist(point_a, point_b)
        self.assertAllClose(expected, result, atol=0, rtol=1e-1)
