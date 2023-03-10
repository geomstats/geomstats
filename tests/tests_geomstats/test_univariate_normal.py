"""Unit tests for the manifold of normal distributions."""

import warnings

from scipy.stats import norm

import geomstats.backend as gs
import tests.conftest
from geomstats.information_geometry.normal import (
    UnivariateNormalDistributions,
    UnivariateNormalMetric,
)


class TestUnivariateNormalDistributions(tests.conftest.TestCase):
    """Class defining the normal distributions tests."""

    def setup_method(self):
        """Define the parameters of the tests."""
        warnings.simplefilter("ignore", category=UserWarning)
        self.normal = UnivariateNormalDistributions()
        self.metric = UnivariateNormalMetric()
        self.n_samples = 10
        self.dim = self.normal.dim

    def test_random_point_and_belongs(self):
        """Test random_point and belongs.

        Test that the random uniform method samples
        on the normal distribution space.
        """
        point = self.normal.random_point()
        result = self.normal.belongs(point)
        expected = True
        self.assertAllClose(expected, result)

    def test_random_point_and_belongs_vectorization(self):
        """Test random_point and belongs.

        Test that the random uniform method samples
        on the normal distribution space.
        """
        n_samples = self.n_samples
        point = self.normal.random_point(n_samples)
        result = self.normal.belongs(point)
        expected = gs.array([True] * n_samples)
        self.assertAllClose(expected, result)

    def test_random_point(self):
        """Test random_point.

        Test that the random uniform method samples points of the right shape
        """
        point = self.normal.random_point(self.n_samples)
        self.assertAllClose(gs.shape(point), (self.n_samples, self.dim))

    def test_sample(self):
        """Test samples."""
        n_points = self.n_samples
        n_samples = 100
        points = self.normal.random_point(n_points)
        samples = self.normal.sample(points, n_samples)
        result = samples.shape
        expected = (n_points, n_samples)

        self.assertAllClose(result, expected)

    def test_point_to_pdf(self):
        """Test point_to_pdf

        Check vectorization of the computation of the pdf.
        """
        point = self.normal.random_point(n_samples=3)
        pdf = self.normal.point_to_pdf(point)
        x = gs.linspace(0.0, 1.0, 10)
        result = pdf(x)
        expected = gs.transpose(
            gs.array([norm.pdf(x_, loc=point[..., 0], scale=point[..., 1]) for x_ in x])
        )
        self.assertAllClose(result, expected)

    def test_normal_metric(self):
        n_samples = 3
        base_point = self.normal.random_point(n_samples)
        vec_a = self.normal.random_tangent_vec(base_point, n_samples)
        vec_b = self.normal.random_tangent_vec(base_point, n_samples)
        mat_prod = gs.einsum(
            "ik,ikj->ij", vec_a, self.normal.metric.metric_matrix(base_point)
        )
        result = gs.einsum("ij,ij->i", mat_prod, vec_b)
        expected = self.normal.metric.inner_product(vec_a, vec_b, base_point)
        self.assertAllClose(result, expected)

    def test_sectional_curvature(self):
        n_samples = 3
        base_point = self.normal.random_point(n_samples)
        vec_a = self.normal.random_tangent_vec(base_point, n_samples)
        vec_b = self.normal.random_tangent_vec(base_point, n_samples)
        result = self.normal.metric.sectional_curvature(vec_a, vec_b, base_point)

        expected = gs.tile(gs.array(-1 / 2), (n_samples,))
        self.assertAllClose(result, expected)
