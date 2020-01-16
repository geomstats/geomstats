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

    def test_product_distance(self):
        point_a_intrinsic = gs.array([0.01, 0.0])
        point_b_intrinsic = gs.array([0.0, 0.0])
        hyperbolic_space_instance = HyperbolicSpace(dimension=2)
        point_a = hyperbolic_space_instance.intrinsic_to_extrinsic_coords(point_a_intrinsic)
        point_b = hyperbolic_space_instance.intrinsic_to_extrinsic_coords(point_b_intrinsic)
        list_points_shape = list(point_a.shape)
        list_dimension_two_disks_data = [2] + list_points_shape
        duplicate_point_a = gs.zeros(list_dimension_two_disks_data)
        duplicate_point_a[0, ...] = point_a
        duplicate_point_a[1, ...] = point_a
        duplicate_point_b = gs.zeros(list_dimension_two_disks_data)
        duplicate_point_b[0, ...] = point_b
        duplicate_point_b[1, ...] = point_b
        single_disk = PoincarePolydisk(n_disks=1)
        two_disks = PoincarePolydisk(n_disks=2)
        distance_single_disk = single_disk.metric.dist(point_a, point_b)
        distance_two_disks = two_disks.metric.dist(duplicate_point_a, duplicate_point_b)
        self.assertAllClose(3 ** 0.5 * distance_single_disk, distance_two_disks)
