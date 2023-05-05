import math
import random

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDBuresWassersteinMetric,
    SPDEuclideanMetric,
    SPDLogEuclideanMetric,
    SPDMatrices,
)
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData

SQRT_2 = math.sqrt(2.0)
LN_2 = math.log(2.0)
EXP_1 = math.exp(1.0)
EXP_2 = math.exp(2.0)
SINH_1 = math.sinh(1.0)


class SPDMatricesTestData(_OpenSetTestData):

    smoke_space_args_list = [(2,), (3,), (4,), (5,)]
    smoke_n_points_list = [1, 2, 1, 2]
    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    shape_list = [(n, n) for n in n_list]
    n_vecs_list = random.sample(range(1, 10), 2)

    Space = SPDMatrices

    def belongs_test_data(self):
        smoke_data = [
            dict(n=2, mat=gs.array([[3.0, -1.0], [-1.0, 3.0]]), expected=True),
            dict(n=2, mat=gs.array([[1.0, 1.0], [2.0, 1.0]]), expected=False),
            dict(
                n=3,
                mat=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=False,
            ),
            dict(
                n=2,
                mat=gs.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]]),
                expected=[True, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def projection_test_data(self):
        smoke_data = [
            dict(
                n=2,
                mat=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[1.0, 0.0], [0.0, 1.0]]),
            ),
            dict(
                n=2,
                mat=gs.array([[-1.0, 0.0], [0.0, -2.0]]),
                expected=gs.array([[gs.atol, 0.0], [0.0, gs.atol]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def logm_test_data(self):
        smoke_data = [
            dict(
                spd_mat=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[0.0, 0.0], [0.0, 0.0]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def cholesky_factor_test_data(self):
        smoke_data = [
            dict(
                n=2,
                spd_mat=gs.array([[[1.0, 2.0], [2.0, 5.0]], [[1.0, 0.0], [0.0, 1.0]]]),
                expected=gs.array([[[1.0, 0.0], [2.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]),
            ),
            dict(
                n=3,
                spd_mat=gs.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
                expected=gs.array(
                    [
                        [SQRT_2, 0.0, 0.0],
                        [0.0, SQRT_2, 0.0],
                        [0.0, 0.0, SQRT_2],
                    ]
                ),
            ),
        ]
        return self.generate_tests(smoke_data)

    def cholesky_factor_belongs_test_data(self):
        list_n = random.sample(range(1, 100), 10)
        n_samples = 10
        random_data = [
            dict(n=n, mat=self.Space(n).random_point(n_samples)) for n in list_n
        ]
        return self.generate_tests([], random_data)

    def differential_cholesky_factor_test_data(self):
        smoke_data = [
            dict(
                n=2,
                tangent_vec=gs.array([[1.0, 1.0], [1.0, 1.0]]),
                base_point=gs.array([[4.0, 2.0], [2.0, 5.0]]),
                expected=gs.array([[1 / 4, 0.0], [3 / 8, 1 / 16]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def differential_power_test_data(self):
        smoke_data = [
            dict(
                power=0.5,
                tangent_vec=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]]
                ),
                expected=gs.array(
                    [
                        [1.0, 1 / 3, 1 / 3],
                        [1 / 3, 0.125, 0.125],
                        [1 / 3, 0.125, 0.125],
                    ]
                ),
            )
        ]
        return self.generate_tests(smoke_data)

    def inverse_differential_power_test_data(self):
        smoke_data = [
            dict(
                power=0.5,
                tangent_vec=gs.array(
                    [
                        [1.0, 1 / 3, 1 / 3],
                        [1 / 3, 0.125, 0.125],
                        [1 / 3, 0.125, 0.125],
                    ]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]]
                ),
                expected=gs.array([[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def differential_log_test_data(self):
        smoke_data = [
            dict(
                tangent_vec=gs.array(
                    [[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]
                ),
                expected=gs.array(
                    [
                        [1.0, 1.0, 2 * LN_2],
                        [1.0, 1.0, 2 * LN_2],
                        [2 * LN_2, 2 * LN_2, 1],
                    ]
                ),
            )
        ]
        return self.generate_tests(smoke_data)

    def inverse_differential_log_test_data(self):
        smoke_data = [
            dict(
                tangent_vec=gs.array(
                    [
                        [1.0, 1.0, 2 * LN_2],
                        [1.0, 1.0, 2 * LN_2],
                        [2 * LN_2, 2 * LN_2, 1],
                    ]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]
                ),
                expected=gs.array([[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]]),
            )
        ]

        return self.generate_tests(smoke_data)

    def differential_exp_test_data(self):
        smoke_data = [
            dict(
                tangent_vec=gs.array(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
                ),
                expected=gs.array(
                    [
                        [EXP_1, EXP_1, SINH_1],
                        [EXP_1, EXP_1, SINH_1],
                        [SINH_1, SINH_1, 1 / EXP_1],
                    ]
                ),
            )
        ]
        return self.generate_tests(smoke_data)

    def inverse_differential_exp_test_data(self):
        smoke_data = [
            dict(
                tangent_vec=gs.array(
                    [
                        [EXP_1, EXP_1, SINH_1],
                        [EXP_1, EXP_1, SINH_1],
                        [SINH_1, SINH_1, 1 / EXP_1],
                    ]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
                ),
                expected=gs.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            )
        ]
        return self.generate_tests(smoke_data)


class SPDAffineMetricTestData(_RiemannianMetricTestData):
    n_list = random.sample(range(2, 5), 2)
    power_affine_list = [1.0, -0.5]

    metric_args_list = [
        {"power_affine": power_affine} for power_affine in power_affine_list
    ]
    shape_list = [(n, n) for n in n_list]
    space_list = [SPDMatrices(n, equip=False) for n in n_list]

    n_points_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = SPDAffineMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(3, equip=False),
                power_affine=0.5,
                tangent_vec_a=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                tangent_vec_b=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]]
                ),
                expected=713 / 144,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(2, equip=False),
                power_affine=1.0,
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[EXP_2, 0.0], [0.0, EXP_2]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(2, equip=False),
                power_affine=1.0,
                point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                base_point=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                expected=gs.array([[-2 * LN_2, 0.0], [0.0, -2 * LN_2]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=10.0)


class SPDBuresWassersteinMetricTestData(_RiemannianMetricTestData):
    n_list = random.sample(range(2, 5), 2)

    shape_list = [(n, n) for n in n_list]
    space_list = [SPDMatrices(n, equip=False) for n in n_list]
    metric_args_list = [{} for _ in n_list]

    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = SPDBuresWassersteinMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(3, equip=False),
                tangent_vec_a=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                tangent_vec_b=gs.array(
                    [[1.0, 2.0, 4.0], [2.0, 3.0, 8.0], [4.0, 8.0, 5.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.5, 0.5], [0.0, 0.5, 1.5]]
                ),
                expected=4.0,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(2, equip=False),
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[4.0, 0.0], [0.0, 4.0]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(2, equip=False),
                point=gs.array([[4.0, 0.0], [0.0, 4.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[2.0, 0.0], [0.0, 2.0]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def squared_dist_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(2, equip=False),
                point_a=[[1.0, 0.0], [0.0, 1.0]],
                point_b=[[2.0, 0.0], [0.0, 2.0]],
                expected=2 + 4 - (2 * 2 * SQRT_2),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=7.0)

    def parallel_transport_test_data(self):
        smoke_data = [dict(space=SPDMatrices(n, equip=False)) for n in self.n_list]
        return self.generate_tests(smoke_data)


class SPDEuclideanMetricTestData(_RiemannianMetricTestData):
    n_list = random.sample(range(2, 5), 2)
    power_euclidean_list = [1.0, -0.5, 0.5, 1.0, 1.0]

    metric_args_list = [
        {"power_euclidean": power_euclidean} for power_euclidean in power_euclidean_list
    ]
    shape_list = [(n, n) for n in n_list]
    space_list = [SPDMatrices(n, equip=False) for n in n_list]

    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = SPDEuclideanMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(3, equip=False),
                power_euclidean=0.5,
                tangent_vec_a=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                tangent_vec_b=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]]
                ),
                expected=3472 / 576,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_domain_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(3, equip=False),
                power_euclidean=1.0,
                tangent_vec=gs.array(
                    [[-1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 1.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
                ),
                expected=gs.array([-3, 1]),
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(2, equip=False),
                power_euclidean=1.0,
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[3.0, 0.0], [0.0, 3.0]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(2, equip=False),
                power_euclidean=1.0,
                point=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[1.0, 0.0], [0.0, 1.0]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def parallel_transport_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(2, equip=False),
                power_euclidean=1.0,
                tangent_vec_a=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                tangent_vec_b=gs.array([[1.0, 0.0], [0.0, 0.5]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=10.0)


class SPDEuclideanMetricPower1TestData(_RiemannianMetricTestData):
    n_list = random.sample(range(2, 5), 2)
    power_euclidean_list = [1.0] * 5

    connection_args_list = metric_args_list = [
        {"power_euclidean": power_euclidean} for power_euclidean in power_euclidean_list
    ]
    shape_list = [(n, n) for n in n_list]
    space_list = [SPDMatrices(n, equip=False) for n in n_list]

    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = n_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = SPDEuclideanMetric


class SPDLogEuclideanMetricTestData(_RiemannianMetricTestData):
    n_list = random.sample(range(2, 4), 2)

    metric_args_list = [{} for _ in n_list]
    shape_list = [(n, n) for n in n_list]
    space_list = [SPDMatrices(n, equip=False) for n in n_list]

    n_points_list = random.sample(range(1, 4), 2)
    n_samples_list = random.sample(range(1, 4), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 4), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = SPDLogEuclideanMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(3, equip=False),
                tangent_vec_a=gs.array(
                    [[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]]
                ),
                tangent_vec_b=gs.array(
                    [[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]
                ),
                expected=5.0 + (4.0 * ((2 * LN_2) ** 2)),
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(2, equip=False),
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[EXP_2, 0.0], [0.0, EXP_2]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(2, equip=False),
                point=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[LN_2, 0.0], [0.0, LN_2]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def dist_test_data(self):
        smoke_data = [
            dict(
                space=SPDMatrices(2, equip=False),
                point_a=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                point_b=gs.array([[EXP_1, 0.0], [0.0, EXP_1]]),
                expected=SQRT_2,
            )
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=10.0)
