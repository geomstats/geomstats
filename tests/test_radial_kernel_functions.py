"""Unit tests for the KNN classifier."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.radial_kernel_functions import \
    biweight_radial_kernel, bump_radial_kernel, cosine_radial_kernel, \
    gaussian_radial_kernel, inverse_multiquadric_radial_kernel, \
    inverse_quadratic_radial_kernel, laplacian_radial_kernel, \
    logistic_radial_kernel, parabolic_radial_kernel, sigmoid_radial_kernel, \
    triangular_radial_kernel, tricube_radial_kernel, triweight_radial_kernel, \
    uniform_radial_kernel


TOLERANCE = 1e-4


class TestRadialKernelFunctions(geomstats.tests.TestCase):
    """Class defining the radial kernel functions tests."""

    def setUp(self):
        """Define the parameters to test."""
        gs.random.seed(1234)
        self.bandwidth = 1
        self.dim = 1
        self.space = Euclidean(dim=self.dim)
        self.distance = self.space.metric.dist

    @geomstats.tests.np_only
    def test_uniform_radial_kernel(self):
        """Test the uniform radial kernel."""
        distance = 0.5
        weight = uniform_radial_kernel(
            distance=distance,
            bandwidth=self.bandwidth)
        result = weight
        expected = 1 / 2
        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_uniform_radial_kernel_bandwidth(self):
        """Test the bandwidth using the uniform radial kernel ."""
        distance = 0.5
        weight = uniform_radial_kernel(
            distance=distance,
            bandwidth=0.25)
        result = weight
        expected = 0
        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_triangular_radial_kernel(self):
        """Test the triangular radial kernel."""
        distance = 1
        bandwidth = 2
        weight = triangular_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = 1 / 2
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_parabolic_radial_kernel(self):
        """Test the parabolic radial kernel."""
        distance = 1
        bandwidth = 2
        weight = parabolic_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = 9 / 16
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_biweight_radial_kernel(self):
        """Test the biweight radial kernel."""
        distance = 1
        bandwidth = 2
        weight = biweight_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = 15 / 16 * (3 / 4) ** 2
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_triweight_radial_kernel(self):
        """Test the triweight radial kernel."""
        distance = 1
        bandwidth = 2
        weight = triweight_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = 35 / 32 * (3 / 4) ** 3
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_tricube_radial_kernel(self):
        """Test the tricube radial kernel."""
        distance = 1
        bandwidth = 2
        weight = tricube_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = 70 / 81 * (7 / 8) ** 3
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_gaussian_radial_kernel(self):
        """Test the gaussian radial kernel."""
        distance = 1
        bandwidth = 2
        weight = gaussian_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.exp(- 1 / 8) / (2 * gs.pi) ** (1 / 2)
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_cosine_radial_kernel(self):
        """Test the cosine radial kernel."""
        distance = 0.5
        weight = cosine_radial_kernel(
            distance=distance,
            bandwidth=self.bandwidth)
        result = weight
        expected = gs.pi / 4 * 2 ** (1 / 2) / 2
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_logistic_radial_kernel(self):
        """Test the logistic radial kernel."""
        distance = 0.5
        weight = logistic_radial_kernel(
            distance=distance,
            bandwidth=self.bandwidth)
        result = weight
        expected = 1 / (gs.exp(1 / 2) + 2 + gs.exp(- 1 / 2))
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_sigmoid_radial_kernel(self):
        """Test the sigmoid radial kernel."""
        distance = 0.5
        weight = sigmoid_radial_kernel(
            distance=distance,
            bandwidth=self.bandwidth)
        result = weight
        expected = 2 / gs.pi / (gs.exp(1 / 2) + gs.exp(- 1 / 2))
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_bump_radial_kernel(self):
        """Test the bump radial kernel."""
        distance = 0.5
        weight = bump_radial_kernel(
            distance=distance,
            bandwidth=self.bandwidth)
        result = weight
        expected = gs.exp(- 1 / (3 / 4))
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_inverse_quadratic_radial_kernel(self):
        """Test the inverse quadratic radial kernel."""
        distance = 0.5
        weight = inverse_quadratic_radial_kernel(
            distance=distance,
            bandwidth=self.bandwidth)
        result = weight
        expected = 4 / 5
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_inverse_multiquadric_radial_kernel(self):
        """Test the inverse multiquadric radial kernel."""
        distance = 0.5
        weight = inverse_multiquadric_radial_kernel(
            distance=distance,
            bandwidth=self.bandwidth)
        result = weight
        expected = 2 / 5 ** (1 / 2)
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_laplacian_radial_kernel(self):
        """Test the Laplacian radial kernel."""
        distance = 0.5
        weight = laplacian_radial_kernel(
            distance=distance,
            bandwidth=self.bandwidth)
        result = weight
        expected = gs.exp(- 1 / 2)
        self.assertAllClose(expected, result, atol=TOLERANCE)
