import random

import geomstats.backend as gs
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_polydisk import (
    PoincarePolydisk,
    PoincarePolydiskMetric,
)
from tests.data_generation import TestData, _OpenSetTestData


class PoincarePolydiskTestData(_OpenSetTestData):

    n_disks_list = random.sample(range(2, 4), 2)
    space_args_list = [(n_disks,) for n_disks in n_disks_list]
    shape_list = [(n_disks, 3) for n_disks in n_disks_list]
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = PoincarePolydisk

    def dimension_test_data(self):
        smoke_data = [dict(n_disks=2, expected=4), dict(n_disks=3, expected=6)]
        return self.generate_tests(smoke_data)


class PoincarePolydiskMetricTestData(TestData):

    n_disks_list = random.sample(range(2, 4), 2)

    shape_list = [(n_disks, 3) for n_disks in n_disks_list]
    space_list = [PoincarePolydisk(n_disks, equip=False) for n_disks in n_disks_list]
    metric_args_list = [{} for _ in n_disks_list]

    n_points_list = random.sample(range(1, 4), 2)
    n_tangent_vecs_list = random.sample(range(1, 4), 2)
    n_points_a_list = random.sample(range(1, 4), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = PoincarePolydiskMetric

    def signature_test_data(self):
        smoke_data = [
            dict(space=PoincarePolydisk(2, equip=False), expected=(4, 0)),
            dict(space=PoincarePolydisk(4, equip=False), expected=(8, 0)),
        ]
        return self.generate_tests(smoke_data)

    def product_distance_test_data(self):
        point_a_intrinsic = gs.array([0.01, 0.0])
        point_b_intrinsic = gs.array([0.0, 0.0])
        hyperbolic_space = Hyperboloid(dim=2, equip=False)
        point_a_extrinsic = hyperbolic_space.from_coordinates(
            point_a_intrinsic, "intrinsic"
        )
        point_b_extrinsic = hyperbolic_space.from_coordinates(
            point_b_intrinsic, "intrinsic"
        )
        smoke_data = [
            dict(
                space=PoincarePolydisk(2, equip=False),
                point_a_extrinsic=point_a_extrinsic,
                point_b_extrinsic=point_b_extrinsic,
            ),
            dict(
                space=PoincarePolydisk(3, equip=False),
                point_a_extrinsic=point_a_extrinsic,
                point_b_extrinsic=point_b_extrinsic,
            ),
        ]
        return self.generate_tests(smoke_data)
