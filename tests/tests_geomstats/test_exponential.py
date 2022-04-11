"""Unit tests for the manifold of exponential distributions."""

import warnings

from scipy.stats import expon

import geomstats.backend as gs
import geomstats.tests
from geomstats.information_geometry.exponential import (
    ExponentialDistributions,
    ExponentialFisherRaoMetric,
)


class TestExponentialDistributions(geomstats.tests.TestCase):
    """Class defining the exponential distributions tests."""

    def setup_method(self):
        """Define the parameters of the tests."""
        warnings.simplefilter("ignore", category=UserWarning)
        self.exponential = ExponentialDistributions()
        self.metric = ExponentialFisherRaoMetric()
        self.n_samples = 10

    def test_projection(self):
        """Test that the projection method sends non-positive points to atol"""
        single_true_point = self.exponential.random_point()
        result = self.exponential.projection(single_true_point)
        expected = single_true_point
        self.assertAllClose(expected, result, atol=1e-8)

        single_false_point = -1
        result = self.exponential.projection(single_false_point)
        expected = gs.atol

        self.assertAllClose(expected, result, atol=1e-8)

        multiple_true_point = self.exponential.random_point(10)
        result = self.exponential.projection(multiple_true_point)
        expected = multiple_true_point

        self.assertAllClose(expected, result, atol=1e-8)

        multiple_false_point = gs.array([-1, 0.5, 0.21, 1, 1.2, 0, 0, 7, 0.3])
        result = self.exponential.projection(multiple_false_point)
        expected = gs.array(
            [
                gs.atol,
                0.5,
                0.21,
                1,
                1.2,
                gs.atol,
                gs.atol,
                7,
                0.3,
            ]
        )

        self.assertAllClose(expected, result, atol=1e-8)

    def test_random_point_and_belongs(self):
        """Test random_point and belongs.

        Test that the random uniform method samples
        on the exponential distribution space.
        """
        point = self.exponential.random_point()
        result = gs.squeeze(self.exponential.belongs(point))
        expected = True
        self.assertAllClose(expected, result)

    def test_random_point_and_belongs_vectorization(self):
        """Test random_point and belongs.

        Test that the random uniform method samples
        on the exponential distribution space.
        """
        n_samples = self.n_samples
        point = self.exponential.random_point(n_samples)
        result = self.exponential.belongs(point)
        expected = gs.array([True] * n_samples)
        self.assertAllClose(expected, result)

    def test_random_point(self):
        """Test random_point.

        Test that the random uniform method samples points of the right shape
        """
        point = self.exponential.random_point(self.n_samples)
        self.assertAllClose(gs.shape(point), (self.n_samples,))

    def test_sample(self):
        """Test samples."""
        n_points = self.n_samples
        n_samples = 100
        points = self.exponential.random_point(n_points)
        samples = self.exponential.sample(points, n_samples)
        result = samples.shape
        expected = (n_points, n_samples)

        self.assertAllClose(result, expected)

    def test_point_to_pdf(self):
        """Test point_to_pdf

        Check vectorization of the computation of the pdf.
        """
        point = self.exponential.random_point(n_samples=2)
        pdf = self.exponential.point_to_pdf(point)
        x = gs.cast(gs.linspace(0, 1, 11), dtype=gs.float32)
        result = pdf(x)
        pdf1 = expon.pdf(x, loc=0, scale=point[0])
        pdf2 = expon.pdf(x, loc=0, scale=point[1])
        expected = gs.stack([gs.array(pdf1), gs.array(pdf2)], axis=1)

        self.assertAllClose(result, expected, atol=1e-8)
