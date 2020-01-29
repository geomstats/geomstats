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
        self.n_samples = 5
        self.dimension = self.beta.dimension

    @geomstats.tests.np_and_pytorch_only
    def test_random_uniform_and_belongs(self):
        """
        Test that the random uniform method samples
        on the hypersphere space.
        """
        n_samples = self.n_samples
        point = self.beta.random_uniform(n_samples)
        result = self.beta.belongs(point)
        expected = gs.array([True] * n_samples)

        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_pytorch_only
    def test_random_uniform(self):
        point = self.beta.random_uniform(self.n_samples)
        self.assertAllClose(gs.shape(point), (self.n_samples, self.dimension))

    @geomstats.tests.np_only
    def test_log_and_exp_general_case(self):
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
