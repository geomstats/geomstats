"""Unit tests for the multinomial manifold."""

import geomstats.backend as gs
from tests.conftest import Parametrizer, np_backend, pytorch_backend, tf_backend
from tests.data.multinomial_data import MultinomialMetricTestData, MultinomialTestData
from tests.geometry_test_cases import LevelSetTestCase, RiemannianMetricTestCase

TF_OR_PYTORCH_BACKEND = tf_backend() or pytorch_backend()

NOT_AUTOGRAD = tf_backend() or pytorch_backend() or np_backend()


class TestMultinomialDistributions(LevelSetTestCase, metaclass=Parametrizer):
    """Class defining the multinomial distributions tests."""

    skip_test_extrinsic_after_intrinsic = True
    skip_test_intrinsic_after_extrinsic = True
    testing_data = MultinomialTestData()

    def test_sample_shape(self, dim, n_draws, point, n_samples, expected):
        self.assertAllClose(
            self.Space(dim, n_draws).sample(point, n_samples).shape, expected
        )

    def test_projection(self, dim, n_draws):
        """Test projection.
        Test that result belongs to the simplex.
        """
        n_points = 4
        points = -10 + 20 * gs.random.rand(n_points, dim + 1)
        projected_points = self.Space(dim, n_draws).projection(points)
        result = gs.sum(projected_points, -1)
        expected = gs.ones(n_points)
        self.assertAllClose(expected, result)

    def test_to_tangent(self, dim, n_draws):
        """Test to_tangent.
        Test that the result belongs to the tangent space to
        the simplex.
        """
        n_points = 4
        vectors = -5 + 2 * gs.random.rand(n_points, dim + 1)
        projected_vectors = self.Space(dim, n_draws).to_tangent(vectors)
        result = gs.sum(projected_vectors, -1)
        expected = gs.zeros(n_points)
        self.assertAllClose(expected, result, atol=1e-05)


class TestMultinomialMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_log_after_exp = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_ivp_belongs = tf_backend()
    skip_test_geodesic_bvp_belongs = tf_backend()
    skip_test_exp_geodesic_ivp = True
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

    def test_simplex_to_sphere_and_back(self, dim, n_draws, n_points):
        """Test simplex_to_sphere and sphere_to_simplex.
        Check that they are inverse.
        """
        points = self.Space(dim, n_draws).random_point(n_points)
        points_sphere = self.Metric(dim, n_draws).simplex_to_sphere(points)
        result = self.Metric(dim, n_draws).sphere_to_simplex(points_sphere)
        expected = points
        self.assertAllClose(expected, result)

    def test_tangent_simplex_to_sphere_and_back(self, dim, n_draws, n_points):
        """Test tangent_simplex_to_sphere and back.
        Check that they are inverse.
        """
        points = self.Space(dim, n_draws).random_point(n_points)
        points_sphere = self.Metric(dim, n_draws).simplex_to_sphere(points)
        vec = -5 + 2 * gs.random.rand(n_points, dim + 1)
        tangent_vec = self.Space(dim, n_draws).to_tangent(vec)
        tangent_vec_sphere = self.Metric(dim, n_draws).tangent_simplex_to_sphere(
            tangent_vec, points
        )
        result = self.Metric(dim, n_draws).tangent_sphere_to_simplex(
            tangent_vec_sphere, points_sphere
        )
        expected = tangent_vec
        self.assertAllClose(expected, result)

    def test_tangent_simplex_to_sphere_vectorization(self, dim, n_draws, n_points):
        """Test tangent_simplex_to_sphere vectorization.
        Check with one point and several tangent vectors.
        """
        point = self.Space(dim, n_draws).random_point()
        point_sphere = self.Metric(dim, n_draws).simplex_to_sphere(point)
        vec = -5 + 2 * gs.random.rand(n_points, dim + 1)
        tangent_vec = self.Space(dim, n_draws).to_tangent(vec)
        tangent_vec_sphere = self.Metric(dim, n_draws).tangent_simplex_to_sphere(
            tangent_vec, point
        )
        result = self.Metric(dim, n_draws).tangent_sphere_to_simplex(
            tangent_vec_sphere, point_sphere
        )
        expected = tangent_vec
        self.assertAllClose(expected, result)

    def test_geodesic(self, dim, n_draws):
        """Test geodesic.
        Check that the norm of the velocity is constant.
        """
        initial_point = self.Space(dim, n_draws).random_point()
        end_point = self.Space(dim, n_draws).random_point()

        n_steps = 100
        geod = self.Metric(dim, n_draws).geodesic(
            initial_point=initial_point, end_point=end_point
        )
        t = gs.linspace(0.0, 1.0, n_steps)
        geod_at_t = geod(t)
        velocity = n_steps * (geod_at_t[1:, :] - geod_at_t[:-1, :])
        velocity_norm = self.Metric(dim, n_draws).norm(velocity, geod_at_t[:-1, :])
        result = (
            1
            / gs.amin(velocity_norm)
            * (gs.amax(velocity_norm) - gs.amin(velocity_norm))
        )
        expected = 0.0

        self.assertAllClose(expected, result, rtol=1.0)

    def test_geodesic_vectorization(self, dim, n_draws, n_points):
        """Check vectorization of geodesic.
        Check the shape of geodesic at time t for
        different scenarios.
        """
        initial_point = self.Space(dim, n_draws).random_point()
        vec = self.Space(dim, n_draws).random_point()
        initial_tangent_vec = self.Space(dim, n_draws).to_tangent(vec)
        geod = self.Metric(dim, n_draws).geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )
        time = 0.5
        result = geod(time).shape
        expected = (dim + 1,)
        self.assertAllClose(expected, result)

        n_vecs = 5
        n_times = 10
        vecs = self.Space(dim, n_draws).random_point(n_vecs)
        initial_tangent_vecs = self.Space(dim, n_draws).to_tangent(vecs)
        geod = self.Metric(dim, n_draws).geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vecs
        )
        times = gs.linspace(0.0, 1.0, n_times)
        result = geod(times).shape
        expected = (n_vecs, n_times, dim + 1)
        self.assertAllClose(result, expected)

        end_points = self.Space(dim, n_draws).random_point(n_points)
        geod = self.Metric(dim, n_draws).geodesic(
            initial_point=initial_point, end_point=end_points
        )
        time = 0.5
        result = geod(time).shape
        expected = (n_points, dim + 1)
        self.assertAllClose(expected, result)

    def test_sectional_curvature_is_positive(self, dim, n_draws, base_point):
        space = self.Space(dim, n_draws)
        metric = self.Metric(dim, n_draws)
        tangent_vec = space.to_tangent(gs.random.rand(dim + 1), base_point)
        result = metric.sectional_curvature(tangent_vec, tangent_vec, base_point)
        self.assertAllClose(gs.all(result > 0), True)
