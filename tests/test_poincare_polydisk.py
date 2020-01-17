"""
Unit tests for the Poincare Polydisk.
"""


import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry.hyperbolic_space import HyperbolicSpace
from geomstats.geometry.poincare_polydisk import PoincarePolydisk


class TestPoincarePolydiskMethods(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)
        self.n_disks = 5
        self.space = PoincarePolydisk(n_disks=self.n_disks)
        self.metric = self.space.metric

    def test_dimension(self):
        expected = self.n_disks * 2
        result = self.space.dimension
        self.assertAllClose(result, expected)

    def test_metric_signature(self):
        expected = (self.n_disks * 2, 0, 0)
        result = self.metric.signature
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_product_distance(self):
        point_type = 'ball'
        point_a = gs.array([0.01, 0.0])
        point_b = gs.array([0.0, 0.0])
        duplicate_point_a = gs.zeros((2,) + point_a.shape)
        duplicate_point_a[0] = point_a
        duplicate_point_a[1] = point_a
        duplicate_point_b = gs.zeros((2,) + point_b.shape)
        duplicate_point_b[0] = point_b
        duplicate_point_b[1] = point_b
        single_disk = PoincarePolydisk(n_disks=1, point_type=point_type)
        two_disks = PoincarePolydisk(n_disks=2, point_type=point_type)
        distance_single_disk = single_disk.metric.dist(point_a, point_b)
        distance_two_disks = two_disks.metric.dist(
            duplicate_point_a, duplicate_point_b)
        result = distance_two_disks
        expected = 3 ** 0.5 * distance_single_disk
        self.assertAllClose(result, expected)

    def test_product_distance_extrinsic_representation(self):
        point_type = 'extrinsic'
        point_a_intrinsic = gs.array([0.01, 0.0])
        point_b_intrinsic = gs.array([0.0, 0.0])
        hyperbolic_space = HyperbolicSpace(dimension=2, point_type=point_type)
        point_a = hyperbolic_space.intrinsic_to_extrinsic_coords(
            point_a_intrinsic)
        point_b = hyperbolic_space.intrinsic_to_extrinsic_coords(
            point_b_intrinsic)
        duplicate_point_a = gs.zeros((2,) + point_a.shape)
        duplicate_point_a[0] = point_a
        duplicate_point_a[1] = point_a
        duplicate_point_b = gs.zeros((2,) + point_b.shape)
        duplicate_point_b[0] = point_b
        duplicate_point_b[1] = point_b
        single_disk = PoincarePolydisk(n_disks=1, point_type=point_type)
        two_disks = PoincarePolydisk(n_disks=2, point_type=point_type)
        distance_single_disk = single_disk.metric.dist(point_a, point_b)
        distance_two_disks = two_disks.metric.dist(
            duplicate_point_a, duplicate_point_b)
        result = distance_two_disks
        expected = 3 ** 0.5 * distance_single_disk
        self.assertAllClose(result, expected)
