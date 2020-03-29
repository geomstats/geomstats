"""Unit tests for the beta manifold."""

import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.beta_distributions import BetaDistributions
from geomstats.geometry.beta_distributions import BetaMetric


class TestBetaMethods(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)
        self.beta = BetaDistributions()
        self.metric = BetaMetric()
        self.n_samples = 10
        self.dimension = self.beta.dimension

    @geomstats.tests.np_and_pytorch_only
    def test_random_uniform_and_belongs(self):
        """
        Test that the random uniform method samples
        on the beta distribution space.
        """
        n_samples = self.n_samples
        point = self.beta.random_uniform(n_samples)
        result = self.beta.belongs(point)
        expected = gs.array([True] * n_samples)

        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_pytorch_only
    def test_random_uniform(self):
        """
        Test that the random uniform method samples points of the right shape
        """
        point = self.beta.random_uniform(self.n_samples)
        self.assertAllClose(gs.shape(point), (self.n_samples, self.dimension))

    @geomstats.tests.np_only
    def test_sample(self):
        """
        Test that the sample method samples variates from beta distributions
        with the specified parameters, using the law of large numbers
        """
        n_samples = self.n_samples
        tol = (n_samples * 10) ** (- 0.5)
        point = self.beta.random_uniform(n_samples)
        samples = self.beta.sample(point, n_samples * 10)
        result = gs.mean(samples, axis=1)
        expected = point[:, 0] / gs.sum(point, axis=1)

        self.assertAllClose(result, expected, rtol=tol, atol=tol)

    @geomstats.tests.np_only
    def test_maximum_likelihood_fit(self):
        """
        Test that the maximum likelihood fit method recovers
        parameters of beta distribution.
        """
        n_samples = self.n_samples
        point = self.beta.random_uniform(n_samples)
        samples = self.beta.sample(point, n_samples * 10)
        fits = self.beta.maximum_likelihood_fit(samples)
        expected = self.beta.belongs(fits)
        result = gs.array([True] * n_samples)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_exp(self):
        gs.random.seed(123)
        n_samples = self.n_samples
        points = self.beta.random_uniform(n_samples)
        vectors = self.beta.random_uniform(n_samples)
        initial_vectors = gs.array([[vec_x, vec_x] for vec_x in vectors[:, 0]])
        points = gs.array([[param_a, param_a] for param_a in points[:, 0]])
        result_points = self.metric.exp(initial_vectors, points)
        result = gs.isclose(result_points[:, 0], result_points[:, 1]).all()
        expected = gs.array([True] * n_samples)

        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_log_and_exp(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        n_samples = self.n_samples
        gs.random.seed(123)
        base_point = self.beta.random_uniform(n_samples=n_samples, bound=5)
        point = self.beta.random_uniform(n_samples=n_samples, bound=5)
        log = self.metric.log(point, base_point, n_steps=500)
        expected = point
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        self.assertAllClose(result, expected, rtol=1e-2)

    def test_christoffels_vectorization(self):
        """
        Check vectorization of Christoffel symbols in
        spherical coordinates on the 2-sphere.
        """
        points = self.beta.random_uniform(self.n_samples)
        christoffel = self.metric.christoffels(points)
        result = christoffel.shape
        expected = gs.array(
            [self.n_samples, self.dimension, self.dimension, self.dimension])
        self.assertAllClose(result, expected)
