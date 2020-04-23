"""Unit tests for the Poincare Polydisk."""


import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_polydisk import PoincarePolydisk


class TestPoincarePolydisk(geomstats.tests.TestCase):
    """Class defining the Poincare polydisk tests."""

    def setUp(self):
        """Define the elements to test."""
        gs.random.seed(1234)
        self.n_disks = 5
        self.space = PoincarePolydisk(n_disks=self.n_disks)
        self.metric = self.space.metric

    def test_dimension(self):
        """Test the dimension."""
        expected = self.n_disks * 2
        result = self.space.dim
        self.assertAllClose(result, expected)

    def test_metric_signature(self):
        """Test the signature."""
        expected = (self.n_disks * 2, 0, 0)
        result = self.metric.signature
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_product_distance_extrinsic_representation(self):
        """Test the distance using the extrinsic representation."""
        coords_type = 'extrinsic'
        point_a_intrinsic = gs.array([0.01, 0.0])
        point_b_intrinsic = gs.array([0.0, 0.0])
        hyperbolic_space = Hyperboloid(dim=2)
        point_a = hyperbolic_space.from_coordinates(
            point_a_intrinsic, 'intrinsic')
        point_b = hyperbolic_space.from_coordinates(
            point_b_intrinsic, 'intrinsic')

        duplicate_point_a = gs.stack([point_a, point_a], axis=0)
        duplicate_point_b = gs.stack([point_b, point_b], axis=0)

        single_disk = PoincarePolydisk(n_disks=1, coords_type=coords_type)
        two_disks = PoincarePolydisk(n_disks=2, coords_type=coords_type)

        distance_single_disk = single_disk.metric.dist(point_a, point_b)
        distance_two_disks = two_disks.metric.dist(
            duplicate_point_a, duplicate_point_b)
        result = distance_two_disks
        expected = 3 ** 0.5 * distance_single_disk
        self.assertAllClose(result, expected)
