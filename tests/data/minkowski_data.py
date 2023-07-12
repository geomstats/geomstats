import math
import random

import geomstats.backend as gs
from geomstats.geometry.minkowski import Minkowski, MinkowskiMetric
from tests.data_generation import _RiemannianMetricTestData, _VectorSpaceTestData


class MinkowskiTestData(_VectorSpaceTestData):
    Space = Minkowski

    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    shape_list = space_args_list
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    def belongs_test_data(self):
        smoke_data = [dict(dim=2, point=gs.array([-1.0, 3.0]), expected=True)]
        return self.generate_tests(smoke_data)


class MinkowskiMetricTestData(_RiemannianMetricTestData):
    n_list = random.sample(range(2, 4), 2)

    shape_list = [(n,) for n in n_list]
    space_list = [Minkowski(n) for n in n_list]
    metric_args_list = [{} for _ in n_list]

    n_points_list = random.sample(range(1, 3), 2)
    n_tangent_vecs_list = random.sample(range(1, 3), 2)
    n_points_a_list = random.sample(range(1, 3), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = MinkowskiMetric

    def metric_matrix_test_data(self):
        space = Minkowski(2, equip=False)
        smoke_data = [dict(space=space, expected=gs.array([[-1.0, 0.0], [0.0, 1.0]]))]
        return self.generate_tests(smoke_data)

    def inner_product_test_data(self):
        space = Minkowski(2, equip=False)
        smoke_data = [
            dict(
                space=space,
                point_a=gs.array([0.0, 1.0]),
                point_b=gs.array([2.0, 10.0]),
                expected=10.0,
            ),
            dict(
                space=space,
                point_a=gs.array([[-1.0, 0.0], [1.0, 0.0], [2.0, math.sqrt(3)]]),
                point_b=gs.array(
                    [
                        [2.0, -math.sqrt(3)],
                        [4.0, math.sqrt(15)],
                        [-4.0, math.sqrt(15)],
                    ]
                ),
                expected=gs.array([2.0, -4.0, 14.70820393]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def squared_norm_test_data(self):
        smoke_data = [
            dict(
                space=Minkowski(2, equip=False),
                point=gs.array([-2.0, 4.0]),
                expected=12.0,
            )
        ]
        return self.generate_tests(smoke_data)

    def squared_dist_test_data(self):
        smoke_data = [
            dict(
                space=Minkowski(2, equip=False),
                point_a=gs.array([2.0, -math.sqrt(3)]),
                point_b=gs.array([4.0, math.sqrt(15)]),
                expected=27.416407,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                space=Minkowski(2, equip=False),
                tangent_vec=gs.array([2.0, math.sqrt(3)]),
                base_point=gs.array([1.0, 0.0]),
                expected=gs.array([3.0, math.sqrt(3)]),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                space=Minkowski(2, equip=False),
                point=gs.array([2.0, math.sqrt(3)]),
                base_point=gs.array([-1.0, 0.0]),
                expected=gs.array([3.0, math.sqrt(3)]),
            )
        ]
        return self.generate_tests(smoke_data)
