"""Unit tests for the Poincare Polydisk."""
import random

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_polydisk import (
    PoincarePolydisk,
    PoincarePolydiskMetric,
)
from tests.conftest import Parametrizer, TestCase
from tests.data_generation import TestData, _OpenSetTestData
from tests.geometry_test_cases import OpenSetTestCase


class TestPoincarePolydisk(OpenSetTestCase, metaclass=Parametrizer):
    space = PoincarePolydisk
    skip_test_to_tangent_is_tangent = True
    skip_test_to_tangent_is_tangent_in_ambient_space = True

    class PoincarePolydiskTestData(_OpenSetTestData):

        n_disks_list = random.sample(range(2, 4), 2)
        space_args_list = [(n_disks,) for n_disks in n_disks_list]
        shape_list = [(n_disks, 3) for n_disks in n_disks_list]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def dimension_test_data(self):
            smoke_data = [dict(n_disks=2, expected=4), dict(n_disks=3, expected=6)]
            return self.generate_tests(smoke_data)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(2,), (3,)]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
                belongs_atol=1e-3,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list,
                self.shape_list,
                self.n_samples_list,
                belongs_atol=1e-1,
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                PoincarePolydisk,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
                is_tangent_atol=1e-3,
            )

        def to_tangent_is_tangent_in_ambient_space_test_data(self):
            return self._to_tangent_is_tangent_in_ambient_space_test_data(
                PoincarePolydisk, self.space_args_list, self.shape_list
            )

    testing_data = PoincarePolydiskTestData()

    def test_dimension(self, n_disks, expected):
        space = PoincarePolydisk(n_disks)
        self.assertAllClose(space.dim, expected)


class TestPoincarePolydiskMetric(TestCase, metaclass=Parametrizer):
    metric = connection = PoincarePolydiskMetric

    class PoincarePolydiskMetricTestData(TestData):

        n_disks_list = random.sample(range(2, 5), 2)
        metric_args_list = [(n_disks,) for n_disks in n_disks_list]
        shape_list = [(n_disks, 3) for n_disks in n_disks_list]
        space_list = [PoincarePolydisk(n_disks) for n_disks in n_disks_list]
        n_points_list = random.sample(range(1, 7), 5)
        n_samples_list = random.sample(range(1, 7), 5)
        n_points_a_list = random.sample(range(1, 7), 5)
        n_points_b_list = [1]
        batch_size_list = random.sample(range(2, 7), 5)
        alpha_list = [1] * 5
        n_rungs_list = [1] * 5
        scheme_list = ["pole"] * 5

        def signature_test_data(self):
            smoke_data = [
                dict(n_disks=2, expected=(4, 0)),
                dict(n_disks=4, expected=(8, 0)),
            ]
            return self.generate_tests(smoke_data)

        def product_distance_extrinsic_representation_test_data(self):
            point_a_intrinsic = gs.array([0.01, 0.0])
            point_b_intrinsic = gs.array([0.0, 0.0])
            hyperbolic_space = Hyperboloid(dim=2)
            point_a_extrinsic = hyperbolic_space.from_coordinates(
                point_a_intrinsic, "intrinsic"
            )
            point_b_extrinsic = hyperbolic_space.from_coordinates(
                point_b_intrinsic, "intrinsic"
            )
            smoke_data = [
                dict(
                    n_disks=1,
                    point_a_extrinsic=point_a_extrinsic,
                    point_b_extrinsic=point_b_extrinsic,
                )
            ]
            return self.generate_tests(smoke_data)

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
