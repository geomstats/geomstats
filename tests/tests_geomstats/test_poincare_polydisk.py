"""Unit tests for the Poincare Polydisk."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.poincare_polydisk import (
    PoincarePolydisk,
    PoincarePolydiskMetric,
)
from tests.conftest import Parametrizer, TestCase
from tests.data.poincare_polydisk_data import (
    PoincarePolydiskMetricTestData,
    PoincarePolydiskTestData,
)
from tests.geometry_test_cases import OpenSetTestCase


class TestPoincarePolydisk(OpenSetTestCase, metaclass=Parametrizer):

    skip_test_to_tangent_is_tangent_in_ambient_space = True
    skip_test_to_tangent_is_tangent = True

    testing_data = PoincarePolydiskTestData()
    space = testing_data.space

    def test_dimension(self, n_disks, expected):
        space = self.space(n_disks)
        self.assertAllClose(space.dim, expected)


class TestPoincarePolydiskMetric(TestCase, metaclass=Parametrizer):
    metric = connection = PoincarePolydiskMetric

    testing_data = PoincarePolydiskMetricTestData()

    def test_signature(self, n_disks, expected):
        metric = PoincarePolydiskMetric(n_disks)
        self.assertAllClose(metric.signature, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_product_distance_extrinsic_representation(
        self, n_disks, point_a_extrinsic, point_b_extrinsic
    ):
        duplicate_point_a = gs.stack([point_a_extrinsic, point_a_extrinsic], axis=0)
        duplicate_point_b = gs.stack([point_b_extrinsic, point_b_extrinsic], axis=0)

        single_disk = PoincarePolydisk(n_disks=n_disks, coords_type="extrinsic")
        two_disks = PoincarePolydisk(n_disks=2 * n_disks, coords_type="extrinsic")

        distance_single_disk = single_disk.metric.dist(
            point_a_extrinsic[None, :], point_b_extrinsic[None, :]
        )
        distance_two_disks = two_disks.metric.dist(duplicate_point_a, duplicate_point_b)
        result = distance_two_disks
        expected = 3**0.5 * distance_single_disk
        self.assertAllClose(result, expected)
