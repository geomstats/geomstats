import math
import random

import geomstats.backend as gs
from geomstats.geometry.siegel import Siegel, SiegelMetric
from tests.data_generation import _ComplexRiemannianMetricTestData, _OpenSetTestData

SQRT_2 = math.sqrt(2.0)
LN_2 = math.log(2.0)
EXP_1 = math.exp(1.0)
EXP_2 = math.exp(2.0)
SINH_1 = math.sinh(1.0)


class SiegelTestData(_OpenSetTestData):

    smoke_space_args_list = [(2,), (3,), (4,), (5,)]
    smoke_n_points_list = [1, 2, 1, 2]
    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    shape_list = [(n, n) for n in n_list]
    n_vecs_list = random.sample(range(1, 10), 2)

    Space = Siegel

    def belongs_test_data(self):
        smoke_data = [
            dict(n=2, mat=[[3.0, -1.0], [-1.0, 3.0]], expected=True),
            dict(n=2, mat=[[1.0, 1.0], [2.0, 1.0]], expected=False),
            dict(
                n=3,
                mat=[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                expected=False,
            ),
            dict(
                n=2,
                mat=[[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]],
                expected=[True, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def projection_test_data(self):
        smoke_data = [
            dict(n=2, mat=[[1.0, 0.0], [0.0, 1.0]], expected=[[1.0, 0.0], [0.0, 1.0]]),
            dict(
                n=2,
                mat=[[-1.0, 0.0], [0.0, -2.0]],
                expected=[[gs.atol, 0.0], [0.0, gs.atol]],
            ),
        ]
        return self.generate_tests(smoke_data)


class SiegelMetricTestData(_ComplexRiemannianMetricTestData):
    n_list = random.sample(range(2, 5), 2)
    scale_list = [1.0, 2]
    metric_args_list = list(zip(n_list, scale_list))
    shape_list = [(n, n) for n in n_list]
    space_list = [Siegel(n) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = SiegelMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                n=3,
                scale=0.5,
                tangent_vec_a=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                tangent_vec_b=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                base_point=[[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]],
                expected=713 / 144,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                n=2,
                scale=1.0,
                tangent_vec=[[2.0, 0.0], [0.0, 2.0]],
                base_point=[[1.0, 0.0], [0.0, 1.0]],
                expected=[[EXP_2, 0.0], [0.0, EXP_2]],
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                n=2,
                scale=1.0,
                point=[[1.0, 0.0], [0.0, 1.0]],
                base_point=[[2.0, 0.0], [0.0, 2.0]],
                expected=[[-2 * LN_2, 0.0], [0.0, -2 * LN_2]],
            )
        ]
        return self.generate_tests(smoke_data)
