import math
import random

import geomstats.backend as gs
from geomstats.geometry.positive_reals import PositiveReals, PositiveRealsMetric
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData

LN_2 = math.log(2.0)
EXP_1 = math.exp(1.0)
EXP_2 = math.exp(2.0)


class PositiveRealsTestData(_OpenSetTestData):

    smoke_space_args_list = [(1,), (1,), (1,), (1,)]
    smoke_n_points_list = [1, 2, 1, 2]
    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    shape_list = [(1,) for _ in n_list]
    n_vecs_list = random.sample(range(1, 10), 2)

    Space = PositiveReals

    def belongs_test_data(self):
        smoke_data = [
            dict(point=[[10.0]], expected=[True]),
            dict(point=[[10 + 0j]], expected=[True]),
            dict(point=[[10 + 1j]], expected=[False]),
            dict(point=[[-10.0]], expected=[False]),
            dict(
                point=[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                expected=[False, False, False],
            ),
            dict(point=[[1.0], [-1.5]], expected=[True, False]),
        ]
        return self.generate_tests(smoke_data)

    def projection_test_data(self):
        smoke_data = [
            dict(point=[[1.0]], expected=[[1.0]]),
            dict(
                point=[[-1.0]],
                expected=[[gs.atol]],
            ),
        ]
        return self.generate_tests(smoke_data)


class PositiveRealsMetricTestData(_RiemannianMetricTestData):
    n_manifolds = 2
    metric_args_list = list(zip(random.sample(range(2, 5), 2)))
    shape_list = [(1,) for i_manifold in range(n_manifolds)]
    space_list = [PositiveReals() for i_manifold in range(n_manifolds)]
    n_points_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = PositiveRealsMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                tangent_vec_a=gs.array([1.0]),
                tangent_vec_b=gs.array([2.0]),
                base_point=gs.array([3.0]),
                expected=2 / 9,
            ),
            dict(
                tangent_vec_a=gs.array([-2.0]),
                tangent_vec_b=gs.array([3.0]),
                base_point=gs.array([4.0]),
                expected=-3 / 8,
            ),
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                tangent_vec=[[1.0]],
                base_point=[[1.0]],
                expected=[[EXP_1]],
            ),
            dict(
                tangent_vec=[[4.0]],
                base_point=[[2.0]],
                expected=[[2 * EXP_2]],
            ),
            dict(
                tangent_vec=[[1.0], [2.0]],
                base_point=[[1.0]],
                expected=[[EXP_1], [EXP_2]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                point=[[4.0]],
                base_point=[[2.0]],
                expected=[[2 * LN_2]],
            ),
            dict(
                point=[[2.0]],
                base_point=[[4.0]],
                expected=[[-4 * LN_2]],
            ),
            dict(
                point=[[1.0], [2.0]],
                base_point=[[1.0]],
                expected=[[0], [LN_2]],
            ),
        ]
        return self.generate_tests(smoke_data)
