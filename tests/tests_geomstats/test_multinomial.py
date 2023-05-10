"""Unit tests for the multinomial manifold."""

import geomstats.backend as gs
from tests.conftest import Parametrizer, np_backend, pytorch_backend
from tests.data.multinomial_data import MultinomialMetricTestData, MultinomialTestData
from tests.geometry_test_cases import LevelSetTestCase, RiemannianMetricTestCase

PYTORCH_BACKEND = pytorch_backend()

NOT_AUTOGRAD = pytorch_backend() or np_backend()


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
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_riemann_tensor_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True

    testing_data = MultinomialMetricTestData()
    Space = testing_data.Space

    def test_simplex_to_sphere_and_back(self, space, n_points):
        """Test simplex_to_sphere and sphere_to_simplex.
        Check that they are inverse.
        """
        space.equip_with_metric(self.Metric)
        expected = points = space.random_point(n_points)
        points_sphere = space.metric.simplex_to_sphere(points)
        result = space.metric.sphere_to_simplex(points_sphere)
        self.assertAllClose(expected, result)

    def test_tangent_simplex_to_sphere_and_back(self, space, n_points):
        """Test tangent_simplex_to_sphere and back.
        Check that they are inverse.
        """
        space.equip_with_metric(self.Metric)
        points = space.random_point(n_points)
        points_sphere = space.metric.simplex_to_sphere(points)
        vec = -5 + 2 * gs.random.rand(n_points, space.dim + 1)
        expected = tangent_vec = space.to_tangent(vec)
        tangent_vec_sphere = space.metric.tangent_simplex_to_sphere(tangent_vec, points)
        result = space.metric.tangent_sphere_to_simplex(
            tangent_vec_sphere, points_sphere
        )
        self.assertAllClose(expected, result)

    def test_tangent_simplex_to_sphere_vectorization(self, space, n_points):
        """Test tangent_simplex_to_sphere vectorization.
        Check with one point and several tangent vectors.
        """
        space.equip_with_metric(self.Metric)
        point = space.random_point()
        point_sphere = space.metric.simplex_to_sphere(point)
        vec = -5 + 2 * gs.random.rand(n_points, space.dim + 1)
        expected = tangent_vec = space.to_tangent(vec)
        tangent_vec_sphere = space.metric.tangent_simplex_to_sphere(tangent_vec, point)
        result = space.metric.tangent_sphere_to_simplex(
            tangent_vec_sphere, point_sphere
        )
        self.assertAllClose(expected, result)

    def test_geodesic(self, space):
        """Test geodesic.
        Check that the norm of the velocity is constant.
        """
        space.equip_with_metric(self.Metric)

        initial_point = space.random_point()
        end_point = space.random_point()

        n_steps = 100
        geod = space.metric.geodesic(initial_point=initial_point, end_point=end_point)
        t = gs.linspace(0.0, 1.0, n_steps)
        geod_at_t = geod(t)
        velocity = n_steps * (geod_at_t[1:, :] - geod_at_t[:-1, :])
        velocity_norm = space.metric.norm(velocity, geod_at_t[:-1, :])
        result = (
            1
            / gs.amin(velocity_norm)
            * (gs.amax(velocity_norm) - gs.amin(velocity_norm))
        )
        expected = 0.0

        self.assertAllClose(expected, result, rtol=1.0)

    def test_geodesic_vectorization(self, space, n_points):
        """Check vectorization of geodesic.
        Check the shape of geodesic at time t for
        different scenarios.
        """
        space.equip_with_metric(self.Metric)
        dim = space.dim

        initial_point = space.random_point()
        vec = space.random_point()
        initial_tangent_vec = space.to_tangent(vec)
        geod = space.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )
        time = 0.5
        result = geod(time).shape
        expected = (dim + 1,)
        self.assertAllClose(expected, result)

        n_vecs = 5
        n_times = 10
        vecs = space.random_point(n_vecs)
        initial_tangent_vecs = space.to_tangent(vecs)
        geod = space.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vecs
        )
        times = gs.linspace(0.0, 1.0, n_times)
        result = geod(times).shape
        expected = (n_vecs, n_times, dim + 1)
        self.assertAllClose(result, expected)

        end_points = space.random_point(n_points)
        geod = space.metric.geodesic(initial_point=initial_point, end_point=end_points)
        time = 0.5
        result = geod(time).shape
        expected = (n_points, dim + 1)
        self.assertAllClose(expected, result)

    def test_sectional_curvature_is_positive(self, space):
        space.equip_with_metric(self.Metric)
        base_point = space.random_point()

        tangent_vec = space.to_tangent(gs.random.rand(space.dim + 1), base_point)
        result = space.metric.sectional_curvature(tangent_vec, tangent_vec, base_point)
        self.assertAllClose(gs.all(result > 0), True)
