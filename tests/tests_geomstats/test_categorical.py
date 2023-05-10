"""Unit tests for the categorical manifold."""

import warnings

import geomstats.backend as gs
import tests.conftest
from geomstats.information_geometry.categorical import CategoricalDistributions
from geomstats.information_geometry.multinomial import MultinomialDistributions


class TestCategoricalDistributions(tests.conftest.TestCase):
    """Class defining the categorical distributions tests."""

    def setup_method(self):
        """Define the parameters of the tests."""
        gs.random.seed(0)
        warnings.simplefilter("ignore", category=UserWarning)
        self.dim = 3
        self.categorical = CategoricalDistributions(self.dim)
        self.metric = self.categorical.metric
        self.n_points = 10
        self.n_samples = 20

    def test_random_point(self):
        """Test random_point.

        Test that the result belongs to the simplex.
        """
        point = self.categorical.random_point(self.n_points)
        result = gs.sum(point, -1)
        expected = gs.ones(self.n_points)
        self.assertAllClose(expected, result)

    def test_projection(self):
        """Test projection.

        Test that result belongs to the simplex.
        """
        points = -10 + 20 * gs.random.rand(self.n_points, self.dim + 1)
        projected_points = self.categorical.projection(points)
        result = gs.sum(projected_points, -1)
        expected = gs.ones(self.n_points)
        self.assertAllClose(expected, result)

    def test_to_tangent(self):
        """Test to_tangent.

        Test that the result belongs to the tangent space to
        the simplex.
        """
        vectors = -5 + 2 * gs.random.rand(self.n_points, self.dim + 1)
        projected_vectors = self.categorical.to_tangent(vectors)
        result = gs.sum(projected_vectors, -1)
        expected = gs.zeros(self.n_points)
        self.assertAllClose(expected, result, atol=1e-05)

    @tests.conftest.np_and_autograd_only
    def test_sample(self):
        """Test sample.

        Check that the samples have the right shape.
        """
        points = self.categorical.random_point(self.n_points)
        samples = self.categorical.sample(points, self.n_samples)
        result = samples.shape
        expected = (self.n_points, self.n_samples)
        self.assertAllClose(expected, result)

    def test_simplex_to_sphere_and_back(self):
        """Test simplex_to_sphere and sphere_to_simplex.

        Check that they are inverse.
        """
        points = self.categorical.random_point(self.n_points)
        points_sphere = self.metric.simplex_to_sphere(points)
        result = self.metric.sphere_to_simplex(points_sphere)
        expected = points
        self.assertAllClose(expected, result)

    def test_tangent_simplex_to_sphere_and_back(self):
        """Test tangent_simplex_to_sphere and back.

        Check that they are inverse.
        """
        points = self.categorical.random_point(self.n_points)
        points_sphere = self.metric.simplex_to_sphere(points)
        vec = -5 + 2 * gs.random.rand(self.n_points, self.dim + 1)
        tangent_vec = self.categorical.to_tangent(vec)
        tangent_vec_sphere = self.metric.tangent_simplex_to_sphere(tangent_vec, points)
        result = self.metric.tangent_sphere_to_simplex(
            tangent_vec_sphere, points_sphere
        )
        expected = tangent_vec
        self.assertAllClose(expected, result)

    def test_tangent_simplex_to_sphere_vectorization(self):
        """Test tangent_simplex_to_sphere vectorization.

        Check with one point and several tangent vectors.
        """
        point = self.categorical.random_point()
        point_sphere = self.metric.simplex_to_sphere(point)
        vec = -5 + 2 * gs.random.rand(self.n_points, self.dim + 1)
        tangent_vec = self.categorical.to_tangent(vec)
        tangent_vec_sphere = self.metric.tangent_simplex_to_sphere(tangent_vec, point)
        result = self.metric.tangent_sphere_to_simplex(tangent_vec_sphere, point_sphere)
        expected = tangent_vec
        self.assertAllClose(expected, result)

    def test_exp_and_log(self):
        """Test exp and log.

        Check that they are inverse.
        """
        base_points = self.categorical.random_point(self.n_points)
        points = self.categorical.random_point(self.n_points)
        log = self.metric.log(points, base_points)
        result = self.metric.exp(log, base_points)
        expected = points
        self.assertAllClose(expected, result)

    def test_exp_and_log_vectorization(self):
        """Test exp and log vectorization.

        Check with one base_point and several points.
        """
        base_point = self.categorical.random_point()
        points = self.categorical.random_point(self.n_points)
        log = self.metric.log(points, base_point)
        result = self.metric.exp(log, base_point)
        expected = points
        self.assertAllClose(expected, result)

    def test_geodesic(self):
        """Test geodesic.

        Check that the norm of the velocity is constant.
        """
        initial_point = self.categorical.random_point()
        end_point = self.categorical.random_point()

        n_steps = 100
        geod = self.metric.geodesic(initial_point=initial_point, end_point=end_point)
        t = gs.linspace(0.0, 1.0, n_steps)
        geod_at_t = geod(t)
        velocity = n_steps * (geod_at_t[1:, :] - geod_at_t[:-1, :])
        velocity_norm = self.metric.norm(velocity, geod_at_t[:-1, :])
        result = (
            1
            / gs.amin(velocity_norm)
            * (gs.amax(velocity_norm) - gs.amin(velocity_norm))
        )
        expected = 0.0

        self.assertAllClose(expected, result, rtol=1.0)

    def test_geodesic_vectorization(self):
        """Check vectorization of geodesic.

        Check the shape of geodesic at time t for
        different scenarios.
        """
        initial_point = self.categorical.random_point()
        vec = self.categorical.random_point()
        initial_tangent_vec = self.categorical.to_tangent(vec)
        geod = self.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vec
        )
        time = 0.5
        result = geod(time).shape
        expected = (self.dim + 1,)
        self.assertAllClose(expected, result)

        n_vecs = 5
        n_times = 10
        vecs = self.categorical.random_point(n_vecs)
        initial_tangent_vecs = self.categorical.to_tangent(vecs)
        geod = self.metric.geodesic(
            initial_point=initial_point, initial_tangent_vec=initial_tangent_vecs
        )
        times = gs.linspace(0.0, 1.0, n_times)
        result = geod(times).shape
        expected = (n_vecs, n_times, self.dim + 1)
        self.assertAllClose(result, expected)

        end_points = self.categorical.random_point(self.n_points)
        geod = self.metric.geodesic(initial_point=initial_point, end_point=end_points)
        time = 0.5
        result = geod(time).shape
        expected = (self.n_points, self.dim + 1)
        self.assertAllClose(expected, result)

    def test_dist(self):
        """Check distance.

        Check that the distance between two multinomial distributions with
        n_draws is equal to the square root of n_draws times the distance
        of the corresponding categorical distributions (n_draws=1).
        """
        n_draws = 10
        multinomial = MultinomialDistributions(n_draws=n_draws, dim=self.dim)
        point_a = multinomial.random_point()
        point_b = multinomial.random_point()
        result = multinomial.metric.dist(point_a, point_b)
        expected = n_draws ** (1 / 2) * self.categorical.metric.dist(point_a, point_b)
        self.assertAllClose(expected, result)
