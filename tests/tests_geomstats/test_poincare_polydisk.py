"""Unit tests for the Poincare Polydisk."""

import geomstats.backend as gs
from geomstats.geometry.poincare_polydisk import PoincarePolydisk
from tests.conftest import Parametrizer, TestCase
from tests.data.poincare_polydisk_data import (
    PoincarePolydiskMetricTestData,
    PoincarePolydiskTestData,
)
from tests.geometry_test_cases import OpenSetTestCase


class TestPoincarePolydisk(OpenSetTestCase, metaclass=Parametrizer):

    skip_test_to_tangent_is_tangent_in_embedding_space = True
    skip_test_to_tangent_is_tangent = True

    testing_data = PoincarePolydiskTestData()

    def test_dimension(self, n_disks, expected):
        space = self.Space(n_disks)
        self.assertAllClose(space.dim, expected)


class TestPoincarePolydiskMetric(TestCase, metaclass=Parametrizer):

    testing_data = PoincarePolydiskMetricTestData()
    Metric = testing_data.Metric

    def test_signature(self, n_disks, expected):
        metric = self.Metric(n_disks)
        self.assertAllClose(metric.signature, expected)

    def test_product_distance(
        self, m_disks, n_disks, point_a_extrinsic, point_b_extrinsic
    ):
        stacked_point_a = gs.stack([point_a_extrinsic for n in range(n_disks)], axis=0)
        stacked_point_b = gs.stack([point_b_extrinsic for n in range(n_disks)], axis=0)

        single_disk = PoincarePolydisk(n_disks=m_disks)
        multiple_disks = PoincarePolydisk(n_disks=m_disks * n_disks)

        distance_single_disk = single_disk.metric.dist(
            point_a_extrinsic[None, :], point_b_extrinsic[None, :]
        )
        distance_n_disks = multiple_disks.metric.dist(stacked_point_a, stacked_point_b)
        result = distance_n_disks
        expected = (n_disks * (n_disks + 1) / 2) ** 0.5 * distance_single_disk
        self.assertAllClose(result, expected)
