import math
import random

import geomstats.backend as gs
from geomstats.geometry.hpd_matrices import (
    HPDAffineMetric,
    HPDBuresWassersteinMetric,
    HPDEuclideanMetric,
    HPDLogEuclideanMetric,
    HPDMatrices,
)
from tests.data_generation import _ComplexRiemannianMetricTestData, _OpenSetTestData

SQRT_2 = math.sqrt(2.0)
LN_2 = math.log(2.0)
EXP_1 = math.exp(1.0)
EXP_2 = math.exp(2.0)
SINH_1 = math.sinh(1.0)


class HPDMatricesTestData(_OpenSetTestData):

    smoke_space_args_list = [(2,), (3,), (4,), (5,)]
    smoke_n_points_list = [1, 2, 1, 2]
    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    shape_list = [(n, n) for n in n_list]
    n_vecs_list = random.sample(range(1, 10), 2)

    Space = HPDMatrices

    def belongs_test_data(self):
        smoke_data = [
            dict(n=2, mat=[[3.0, -1.0], [-1.0, 3.0]], expected=True),
            dict(n=2, mat=[[3j, -1.0], [-1.0, 3.0]], expected=False),
            dict(n=2, mat=[[1.0, 1.0], [2.0, 1.0]], expected=False),
            dict(
                n=3,
                mat=[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                expected=False,
            ),
            dict(
                n=3,
                mat=[[3.0 + 0j, 0j, 1j], [0j, 4.0 + 0j, 0j], [-0.5j, 0j, 6.0 + 0j]],
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
                mat=[[1.0 + 0.0j, 0.5j], [0.5j, 1.0 + 0.0j]],
                expected=[[1.0, 0.0], [0.0, 1.0]],
            ),
            dict(
                n=2,
                mat=[[-1.0, 0.0], [0.0, -2.0]],
                expected=[[gs.atol, 0.0], [0.0, gs.atol]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def logm_test_data(self):
        smoke_data = [
            dict(hpd_mat=[[1.0, 0.0], [0.0, 1.0]], expected=[[0.0, 0.0], [0.0, 0.0]]),
            dict(
                hpd_mat=[[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
                expected=[[0.0, 0.0], [0.0, 0.0]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def cholesky_factor_test_data(self):
        smoke_data = [
            dict(
                n=2,
                hpd_mat=[[[1.0, 2.0], [2.0, 5.0]], [[1.0, 0.0], [0.0, 1.0]]],
                expected=[[[1.0, 0.0], [2.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
            ),
            dict(
                n=2,
                hpd_mat=[
                    [[1.0 + 0j, 2.0 + 0j], [2.0 + 0j, 5.0 + 0j]],
                    [[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
                ],
                expected=[
                    [[1.0 + 0j, 0], [2.0 + 0j, 1.0 + 0j]],
                    [[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
                ],
            ),
            dict(
                n=3,
                hpd_mat=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
                expected=[
                    [SQRT_2, 0.0, 0.0],
                    [0.0, SQRT_2, 0.0],
                    [0.0, 0.0, SQRT_2],
                ],
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
                tangent_vec=[[1.0, 1.0], [1.0, 1.0]],
                base_point=[[4.0, 2.0], [2.0, 5.0]],
                expected=[[1 / 4, 0.0], [3 / 8, 1 / 16]],
            )
        ]
        return self.generate_tests(smoke_data)

    def differential_power_test_data(self):
        smoke_data = [
            dict(
                power=0.5,
                tangent_vec=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                base_point=[[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]],
                expected=[
                    [1.0, 1 / 3, 1 / 3],
                    [1 / 3, 0.125, 0.125],
                    [1 / 3, 0.125, 0.125],
                ],
            )
        ]
        return self.generate_tests(smoke_data)

    def inverse_differential_power_test_data(self):
        smoke_data = [
            dict(
                power=0.5,
                tangent_vec=[
                    [1.0, 1 / 3, 1 / 3],
                    [1 / 3, 0.125, 0.125],
                    [1 / 3, 0.125, 0.125],
                ],
                base_point=[[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]],
                expected=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
            )
        ]
        return self.generate_tests(smoke_data)

    def differential_log_test_data(self):
        smoke_data = [
            dict(
                tangent_vec=[[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]],
                base_point=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]],
                expected=[
                    [1.0, 1.0, 2 * LN_2],
                    [1.0, 1.0, 2 * LN_2],
                    [2 * LN_2, 2 * LN_2, 1],
                ],
            )
        ]
        return self.generate_tests(smoke_data)

    def inverse_differential_log_test_data(self):
        smoke_data = [
            dict(
                tangent_vec=[
                    [1.0, 1.0, 2 * LN_2],
                    [1.0, 1.0, 2 * LN_2],
                    [2 * LN_2, 2 * LN_2, 1],
                ],
                base_point=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]],
                expected=[[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]],
            )
        ]

        return self.generate_tests(smoke_data)

    def differential_exp_test_data(self):
        smoke_data = [
            dict(
                tangent_vec=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                base_point=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                expected=[
                    [EXP_1, EXP_1, SINH_1],
                    [EXP_1, EXP_1, SINH_1],
                    [SINH_1, SINH_1, 1 / EXP_1],
                ],
            )
        ]
        return self.generate_tests(smoke_data)

    def inverse_differential_exp_test_data(self):
        smoke_data = [
            dict(
                tangent_vec=[
                    [EXP_1, EXP_1, SINH_1],
                    [EXP_1, EXP_1, SINH_1],
                    [SINH_1, SINH_1, 1 / EXP_1],
                ],
                base_point=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                expected=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            )
        ]
        return self.generate_tests(smoke_data)


class HPDAffineMetricTestData(_ComplexRiemannianMetricTestData):
    n_list = random.sample(range(2, 5), 2)
    power_affine_list = [1.0, -0.5]
    metric_args_list = list(zip(n_list, power_affine_list))
    shape_list = [(n, n) for n in n_list]
    space_list = [HPDMatrices(n) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = HPDAffineMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                n=3,
                power_affine=0.5,
                tangent_vec_a=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                tangent_vec_b=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                base_point=[[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]],
                expected=713 / 144,
            ),
            dict(
                n=3,
                power_affine=0.5,
                tangent_vec_a=[
                    [2.0 + 0j, 1.0 + 0j, 1.0 + 0j],
                    [1.0 + 0j, 0.5 + 0j, 0.5 + 0j],
                    [1.0 + 0j, 0.5 + 0j, 0.5 + 0j],
                ],
                tangent_vec_b=[
                    [2.0 + 0j, 1.0 + 0j, 1.0 + 0j],
                    [1.0 + 0j, 0.5 + 0j, 0.5 + 0j],
                    [1.0 + 0j, 0.5 + 0j, 0.5 + 0j],
                ],
                base_point=[
                    [1.0 + 0j, 0j, 0],
                    [0, 2.5 + 0j, 1.5 + 0j],
                    [0j, 1.5 + 0j, 2.5 + 0j],
                ],
                expected=713 / 144,
            ),
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                n=2,
                power_affine=1.0,
                tangent_vec=[[2.0, 0.0], [0.0, 2.0]],
                base_point=[[1.0, 0.0], [0.0, 1.0]],
                expected=[[EXP_2, 0.0], [0.0, EXP_2]],
            ),
            dict(
                n=2,
                power_affine=1.0,
                tangent_vec=[[2.0 + 0j, 0j], [0j, 2.0 + 0j]],
                base_point=[[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
                expected=[[EXP_2 + 0j, 0j], [0j, EXP_2 + 0j]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                n=2,
                power_affine=1.0,
                point=[[1.0, 0.0], [0.0, 1.0]],
                base_point=[[2.0, 0.0], [0.0, 2.0]],
                expected=[[-2 * LN_2, 0.0], [0.0, -2 * LN_2]],
            ),
            dict(
                n=2,
                power_affine=1.0,
                point=[[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
                base_point=[[2.0 + 0j, 0j], [0j, 2.0 + 0j]],
                expected=[[-2 * LN_2 + 0j, 0j], [0j, -2 * LN_2 + 0j]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=10.0)


class HPDBuresWassersteinMetricTestData(_ComplexRiemannianMetricTestData):
    n_list = random.sample(range(2, 5), 2)
    metric_args_list = [(n,) for n in n_list]
    shape_list = [(n, n) for n in n_list]
    space_list = [HPDMatrices(n) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = HPDBuresWassersteinMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                n=3,
                tangent_vec_a=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                tangent_vec_b=[[1.0, 2.0, 4.0], [2.0, 3.0, 8.0], [4.0, 8.0, 5.0]],
                base_point=[[1.0, 0.0, 0.0], [0.0, 1.5, 0.5], [0.0, 0.5, 1.5]],
                expected=4.0,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                n=2,
                tangent_vec=[[2.0, 0.0], [0.0, 2.0]],
                base_point=[[1.0, 0.0], [0.0, 1.0]],
                expected=[[4.0, 0.0], [0.0, 4.0]],
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point=[[4.0, 0.0], [0.0, 4.0]],
                base_point=[[1.0, 0.0], [0.0, 1.0]],
                expected=[[2.0, 0.0], [0.0, 2.0]],
            )
        ]
        return self.generate_tests(smoke_data)

    def squared_dist_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point_a=[[1.0, 0.0], [0.0, 1.0]],
                point_b=[[2.0, 0.0], [0.0, 2.0]],
                expected=2 + 4 - (2 * 2 * SQRT_2),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=7.0)

    def parallel_transport_test_data(self):
        smoke_data = [dict(n=k) for k in self.metric_args_list]
        return self.generate_tests(smoke_data)


class HPDEuclideanMetricTestData(_ComplexRiemannianMetricTestData):
    n_list = random.sample(range(2, 5), 2)
    power_euclidean_list = [1.0, -0.5, 0.5, 1.0, 1.0]
    metric_args_list = list(zip(n_list, power_euclidean_list))
    one_metric_args_list = list(zip(n_list, [1.0] * 5))
    shape_list = [(n, n) for n in n_list]
    space_list = [HPDMatrices(n) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = HPDEuclideanMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                n=3,
                power_euclidean=0.5,
                tangent_vec_a=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                tangent_vec_b=[[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
                base_point=[[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]],
                expected=3472 / 576,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_domain_test_data(self):
        smoke_data = [
            dict(
                n=3,
                power_euclidean=1.0,
                tangent_vec=[[-1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 1.0]],
                base_point=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                expected=[-3, 1],
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                n=2,
                power_euclidean=1.0,
                tangent_vec=[[2.0, 0.0], [0.0, 2.0]],
                base_point=[[1.0, 0.0], [0.0, 1.0]],
                expected=[[3.0, 0.0], [0.0, 3.0]],
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                n=2,
                power_euclidean=1.0,
                point=[[2.0, 0.0], [0.0, 2.0]],
                base_point=[[1.0, 0.0], [0.0, 1.0]],
                expected=[[1.0, 0.0], [0.0, 1.0]],
            )
        ]
        return self.generate_tests(smoke_data)

    def parallel_transport_test_data(self):
        smoke_data = [
            dict(
                n=2,
                power_euclidean=1.0,
                tangent_vec_a=[[2.0, 0.0], [0.0, 2.0]],
                base_point=[[1.0, 0.0], [0.0, 1.0]],
                tangent_vec_b=[[1.0, 0.0], [0.0, 0.5]],
            )
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=10.0)


class HPDEuclideanMetricPower1TestData(_ComplexRiemannianMetricTestData):
    n_list = random.sample(range(2, 5), 2)
    power_euclidean_list = [1.0] * 5
    connection_args_list = metric_args_list = list(zip(n_list, [1.0] * 5))
    shape_list = [(n, n) for n in n_list]
    space_list = [HPDMatrices(n) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = n_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = HPDEuclideanMetric


class HPDLogEuclideanMetricTestData(_ComplexRiemannianMetricTestData):

    n_list = random.sample(range(2, 4), 2)
    metric_args_list = [(n,) for n in n_list]
    shape_list = [(n, n) for n in n_list]
    space_list = [HPDMatrices(n) for n in n_list]
    n_points_list = random.sample(range(1, 4), 2)
    n_samples_list = random.sample(range(1, 4), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 4), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = HPDLogEuclideanMetric

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                n=3,
                tangent_vec_a=[[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]],
                tangent_vec_b=[[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]],
                base_point=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]],
                expected=5.0 + (4.0 * ((2 * LN_2) ** 2)),
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                n=2,
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
                point=[[2.0, 0.0], [0.0, 2.0]],
                base_point=[[1.0, 0.0], [0.0, 1.0]],
                expected=[[LN_2, 0.0], [0.0, LN_2]],
            )
        ]
        return self.generate_tests(smoke_data)

    def dist_test_data(self):
        smoke_data = [
            dict(
                n=2,
                point_a=[[1.0, 0.0], [0.0, 1.0]],
                point_b=[[EXP_1, 0.0], [0.0, EXP_1]],
                expected=SQRT_2,
            )
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=10.0)
