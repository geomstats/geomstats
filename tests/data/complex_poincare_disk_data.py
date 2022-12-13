import math
import random

import geomstats.backend as gs
from geomstats.geometry.complex_poincare_disk import (
    ComplexPoincareDisk,
    ComplexPoincareDiskMetric,
)
from tests.data_generation import (
    _ComplexOpenSetTestData,
    _ComplexRiemannianMetricTestData,
)

LN_3 = math.log(3.0)
EXP_4 = math.exp(4.0)
SQRT_2 = math.sqrt(2.0)
SQRT_8 = math.sqrt(8.0)


class ComplexPoincareDiskTestData(_ComplexOpenSetTestData):

    smoke_space_args_list = [(1,), (1,), (1,), (1,)]
    smoke_n_points_list = [1, 2, 1, 2]
    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    shape_list = [(1,) for _ in n_list]
    n_vecs_list = random.sample(range(2, 10), 2)

    Space = ComplexPoincareDisk

    def belongs_test_data(self):
        smoke_data = [
            dict(point=gs.array([0.2 + 0.2j]), expected=gs.array(True)),
            dict(point=gs.array([1.0]), expected=gs.array(False)),
            dict(
                point=gs.array([0.9 + 0.9j]),
                expected=gs.array(False),
            ),
            dict(
                point=gs.array([[0.5 - 0.5j], [-4]]),
                expected=gs.array([True, False]),
            ),
            dict(
                point=gs.array(
                    [
                        [[0.1 + 0.2j, 0.1 + 0.2j], [0.0 - 0.2j, 0.5j]],
                        [[1.0 - 1j, -1.0 + 0.2j], [0.4j, 1.0 + 1.2j]],
                    ]
                ),
                expected=gs.array([[False, False], [False, False]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def projection_test_data(self):
        smoke_data = [
            dict(
                point=gs.array([[1.0], [0.0]]),
                expected=gs.array([[1.0 - gs.atol], [0.0]]),
            ),
            dict(
                point=gs.array([[-1j], [-2.0 + 2j]]),
                expected=gs.array(
                    [
                        [-(1 - gs.atol) * 1j],
                        [(1 - gs.atol) * (-1 + 1j) / 2**0.5],
                    ]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)


class ComplexPoincareDiskMetricTestData(_ComplexRiemannianMetricTestData):

    n_manifolds = 2
    metric_args_list = list(zip(random.sample(range(2, 5), 2)))
    shape_list = [(1,) for i_manifold in range(n_manifolds)]
    space_list = [ComplexPoincareDisk() for i_manifold in range(n_manifolds)]
    n_points_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = ComplexPoincareDiskMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                scale=0.5,
                tangent_vec_a=[[1j]],
                tangent_vec_b=[[1j]],
                base_point=[[0j]],
                expected=[1 / 4],
            ),
            dict(
                scale=1,
                tangent_vec_a=[[3.0 + 4j]],
                tangent_vec_b=[[3.0 + 4j]],
                base_point=[[0.0]],
                expected=[25],
            ),
            dict(
                scale=2,
                tangent_vec_a=[[1j], [1j], [1j]],
                tangent_vec_b=[[1j], [2j], [3j]],
                base_point=[[0j], [0j], [0j]],
                expected=[4, 8, 12],
            ),
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                scale=1.0,
                tangent_vec=[[2.0 + 0j]],
                base_point=[[0.0 + 0j]],
                expected=[[(EXP_4 - 1) / (EXP_4 + 1)]],
            ),
            dict(
                scale=10.0,
                tangent_vec=[[2.0 + 0j]],
                base_point=[[0.0 + 0j]],
                expected=[[(EXP_4 - 1) / (EXP_4 + 1)]],
            ),
            dict(
                scale=1.0,
                tangent_vec=[[2.0 + 2j]],
                base_point=[[0.0 + 0j]],
                expected=[
                    [
                        (1 + 1j)
                        / SQRT_2
                        * (gs.exp(2 * SQRT_8 + 0j) - 1)
                        / (gs.exp(2 * SQRT_8 + 0j) + 1)
                    ]
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                scale=1.0,
                point=[[0.5]],
                base_point=[[0.0]],
                expected=[[LN_3 / 2]],
            )
        ]
        return self.generate_tests(smoke_data)
