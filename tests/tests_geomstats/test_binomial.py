"""Unit tests for the manifold of binomial distributions."""

import warnings

from scipy.stats import binom

import geomstats.backend as gs
import geomstats.tests
from geomstats.information_geometry.binomial import (
    BinomialFisherRaoMetric,
    BinomialDistributions,
)


class TestBinomialDistributions(geomstats.tests.TestCase):
    """Class defining the binomial distributions tests."""

    def setup_method(self):
        """Define the parameters of the tests."""
        warnings.simplefilter("ignore", category=UserWarning)
        self.n_draws = 10
        self.binomial = BinomialDistributions(self.n_draws)
        self.metric = BinomialFisherRaoMetric(self.n_draws)
        self.n_samples = 10

    def test_projection(self):
        """Test that the projection method sends points outside [0,1] to either 0 or 1 (depending on which is closest)"""
        single_true_point = self.binomial.random_point()
        result = self.binomial.projection(single_true_point)
        expected = single_true_point
        self.assertAllClose(expected, result, atol=1e-8)

        single_false_point = -1
        result = self.binomial.projection(single_false_point)
        expected = gs.atol

        self.assertAllClose(expected, result, atol=1e-8)

        multiple_true_point = self.binomial.random_point(10)
        result = self.binomial.projection(multiple_true_point)
        expected = multiple_true_point

        self.assertAllClose(expected, result, atol=1e-8)

        multiple_false_point = gs.array([-1, 0.5, 0.21, 1, 1.2, 0, 0, 7, 0.3])
        result = self.binomial.projection(multiple_false_point)
        expected = gs.array(
            [
                gs.atol,
                0.5,
                0.21,
                1 - gs.atol,
                1 - gs.atol,
                gs.atol,
                gs.atol,
                1 - gs.atol,
                0.3,
            ]
        )
        self.assertAllClose(expected, result, atol=1e-8)

    def test_random_point_and_belongs(self):
        """Test random_point and belongs.

        Test that the random uniform method samples
        on the binomial distribution space.
        """
        point = self.binomial.random_point()
        result = self.binomial.belongs(point)
        expected = True
        self.assertAllClose(expected, result)

    def test_random_point_and_belongs_vectorization(self):
        """Test random_point and belongs.

        Test that the random uniform method samples
        on the binomial distribution space.
        """
        n_samples = self.n_samples
        point = self.binomial.random_point(n_samples)
        result = self.binomial.belongs(point)
        expected = gs.array([True] * n_samples)
        self.assertAllClose(expected, result)

    def test_random_point(self):
        """Test random_point.

        Test that the random uniform method samples points of the right shape
        """
        point = self.binomial.random_point(self.n_samples)
        self.assertAllClose(gs.shape(point), (self.n_samples,))

    def test_sample(self):
        """Test samples."""
        n_points = self.n_samples
        n_samples = 100
        points = self.binomial.random_point(n_points)
        samples = self.binomial.sample(points, n_samples)
        result = samples.shape
        expected = (n_points, n_samples)

        self.assertAllClose(result, expected)

    def test_point_to_pmf(self):
        """Test point_to_pmf

        Check vectorization of the computation of the pmf.
        """
        point = self.binomial.random_point(n_samples=2)
        pmf = self.binomial.point_to_pmf(point)
        k = gs.linspace(0, self.n_draws, self.n_draws + 1)
        result = pmf(k)
        pmf1 = binom.pmf(k, self.n_draws, point[0])
        pmf2 = binom.pmf(k, self.n_draws, point[1])
        expected = gs.stack([gs.array(pmf1), gs.array(pmf2)], axis=1)

        self.assertAllClose(result, expected, atol=1e-8)
