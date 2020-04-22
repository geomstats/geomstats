"""Unit tests for the KNN classifier."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.radial_kernel_functions \
    import cosine_radial_kernel, uniform_radial_kernel

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
    def test_cosine_radial_kernel(self):
        """Test the cosine radial kernel."""
        distance = 0.5
        weight = cosine_radial_kernel(
            distance=distance,
            bandwidth=self.bandwidth)
        result = weight
        expected = gs.pi / 4 * 2 ** (1 / 2) / 2
        self.assertAllClose(expected, result, atol=TOLERANCE)
