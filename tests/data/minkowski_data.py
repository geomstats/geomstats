import math
import random

from geomstats.spaces.equipped import Minkowski
from geomstats.structure.metric import MinkowskiMetric
from tests.data_generation import _RiemannianMetricTestData, _VectorSpaceTestData


class MinkowskiTestData(_VectorSpaceTestData):
    Space = Minkowski

    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    shape_list = space_args_list
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    def belongs_test_data(self):
        smoke_data = [dict(dim=2, point=[-1.0, 3.0], expected=True)]
        return self.generate_tests(smoke_data)


class MinkowskiMetricTestData(_RiemannianMetricTestData):
    Space = Minkowski

    n_list = random.sample(range(2, 4), 2)
    args_list = [(n,) for n in n_list]
    shape_list = args_list
    space_list = [Minkowski(n) for n in n_list]
    n_points_list = random.sample(range(1, 3), 2)
    n_tangent_vecs_list = random.sample(range(1, 3), 2)
    n_points_a_list = random.sample(range(1, 3), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = MinkowskiMetric

    def metric_matrix_test_data(self):
        smoke_data = [dict(dim=2, expected=[[-1.0, 0.0], [0.0, 1.0]])]
        return self.generate_tests(smoke_data)

    def inner_product_test_data(self):
        smoke_data = [
            dict(dim=2, point_a=[0.0, 1.0], point_b=[2.0, 10.0], expected=10.0),
            dict(
                dim=2,
                point_a=[[-1.0, 0.0], [1.0, 0.0], [2.0, math.sqrt(3)]],
                point_b=[
                    [2.0, -math.sqrt(3)],
                    [4.0, math.sqrt(15)],
                    [-4.0, math.sqrt(15)],
                ],
                expected=[2.0, -4.0, 14.70820393],
            ),
        ]
        return self.generate_tests(smoke_data)

    def squared_norm_test_data(self):
        smoke_data = [dict(dim=2, point=[-2.0, 4.0], expected=12.0)]
        return self.generate_tests(smoke_data)

    def squared_dist_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point_a=[2.0, -math.sqrt(3)],
                point_b=[4.0, math.sqrt(15)],
                expected=27.416407,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                tangent_vec=[2.0, math.sqrt(3)],
                base_point=[1.0, 0.0],
                expected=[3.0, math.sqrt(3)],
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point=[2.0, math.sqrt(3)],
                base_point=[-1.0, 0.0],
                expected=[3.0, math.sqrt(3)],
            )
        ]
        return self.generate_tests(smoke_data)
