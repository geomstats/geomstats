"""Unit tests for the radial kernel functions."""

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
        self.dim = 2
        self.space = Euclidean(dim=self.dim)
        self.distance = self.space.metric.dist

    def test_check_distance(self):
        """Test the function checking the distance parameter."""
        distance = gs.array([[1 / 2], [- 2]], dtype=float)
        self.assertRaises(
            ValueError,
            lambda: uniform_radial_kernel(distance=distance))

    def test_check_bandwidth(self):
        """Test the function checking the bandwidth parameter."""
        distance = gs.array([[1 / 2], [2]], dtype=float)
        bandwidth = 0
        self.assertRaises(
            ValueError,
            lambda: uniform_radial_kernel(
                distance=distance,
                bandwidth=bandwidth))

    def test_uniform_radial_kernel(self):
        """Test the uniform radial kernel."""
        distance = gs.array([[1 / 2], [2]], dtype=float)
        weight = uniform_radial_kernel(
            distance=distance,
            bandwidth=self.bandwidth)
        result = weight
        expected = gs.array([[1], [0]])
        self.assertAllClose(expected, result)

    def test_uniform_radial_kernel_bandwidth(self):
        """Test the bandwidth using the uniform radial kernel ."""
        distance = gs.array([[1 / 2], [2]], dtype=float)
        weight = uniform_radial_kernel(
            distance=distance,
            bandwidth=1 / 4)
        result = weight
        expected = gs.array([[0], [0]])
        self.assertAllClose(expected, result)

    def test_triangular_radial_kernel(self):
        """Test the triangular radial kernel."""
        distance = gs.array([[1], [2]], dtype=float)
        bandwidth = 2
        weight = triangular_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.array([[1 / 2], [0]], dtype=float)
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_parabolic_radial_kernel(self):
        """Test the parabolic radial kernel."""
        distance = gs.array([[1], [2]], dtype=float)
        bandwidth = 2
        weight = parabolic_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.array([[3 / 4], [0]], dtype=float)
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_biweight_radial_kernel(self):
        """Test the biweight radial kernel."""
        distance = gs.array([[1], [2]], dtype=float)
        bandwidth = 2
        weight = biweight_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.array([[9 / 16], [0]], dtype=float)
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_triweight_radial_kernel(self):
        """Test the triweight radial kernel."""
        distance = gs.array([[1], [2]], dtype=float)
        bandwidth = 2
        weight = triweight_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.array([[(3 / 4) ** 3], [0]], dtype=float)
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_tricube_radial_kernel(self):
        """Test the tricube radial kernel."""
        distance = gs.array([[1], [2]], dtype=float)
        bandwidth = 2
        weight = tricube_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.array([[(7 / 8) ** 3], [0]], dtype=float)
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_gaussian_radial_kernel(self):
        """Test the gaussian radial kernel."""
        distance = gs.array([[1], [2]], dtype=float)
        bandwidth = 2
        weight = gaussian_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.array(
            [[gs.exp(- 1 / 8)],
             [gs.exp(- 1 / 2)]],
            dtype=float)
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_cosine_radial_kernel(self):
        """Test the cosine radial kernel."""
        distance = gs.array([[1], [2]], dtype=float)
        bandwidth = 2
        weight = cosine_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.array([[2 ** (1 / 2) / 2], [0]], dtype=float)
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_logistic_radial_kernel(self):
        """Test the logistic radial kernel."""
        distance = gs.array([[1], [2]], dtype=float)
        bandwidth = 2
        weight = logistic_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.array(
            [[1 / (gs.exp(1 / 2) + 2 + gs.exp(- 1 / 2))],
             [1 / (gs.exp(1.0) + 2 + gs.exp(- 1.0))]])
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_sigmoid_radial_kernel(self):
        """Test the sigmoid radial kernel."""
        distance = gs.array([[1], [2]], dtype=float)
        bandwidth = 2
        weight = sigmoid_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.array(
            [[1 / (gs.exp(1 / 2) + gs.exp(- 1 / 2))],
             [1 / (gs.exp(1.0) + gs.exp(- 1.0))]],
            dtype=float)
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_bump_radial_kernel(self):
        """Test the bump radial kernel."""
        distance = gs.array([[1 / 2], [2]], dtype=float)
        bandwidth = 1
        weight = bump_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.array([[gs.exp(- 1 / (3 / 4))], [0]], dtype=float)
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_inverse_quadratic_radial_kernel(self):
        """Test the inverse quadratic radial kernel."""
        distance = gs.array([[1], [2]], dtype=float)
        bandwidth = 2
        weight = inverse_quadratic_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.array([[4 / 5], [1 / 2]], dtype=float)
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_inverse_multiquadric_radial_kernel(self):
        """Test the inverse multiquadric radial kernel."""
        distance = gs.array([[1], [2]], dtype=float)
        bandwidth = 2
        weight = inverse_multiquadric_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.array(
            [[2 / 5 ** (1 / 2)],
             [1 / 2 ** (1 / 2)]],
            dtype=float)
        self.assertAllClose(expected, result, atol=TOLERANCE)

    def test_laplacian_radial_kernel(self):
        """Test the Laplacian radial kernel."""
        distance = gs.array([[1], [2]], dtype=float)
        bandwidth = 2
        weight = laplacian_radial_kernel(
            distance=distance,
            bandwidth=bandwidth)
        result = weight
        expected = gs.array(
            [[gs.exp(- 1 / 2)],
             [gs.exp(- 1.0)]],
            dtype=float)
        self.assertAllClose(expected, result, atol=TOLERANCE)
