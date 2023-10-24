import math
import random

import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import OpenSetTestData
from .riemannian_metric import RiemannianMetricTestData

SQRT_2 = math.sqrt(2.0)
LN_2 = math.log(2.0)
EXP_1 = math.exp(1.0)
EXP_2 = math.exp(2.0)
SINH_1 = math.sinh(1.0)


class SPDMatricesMixinsTestData:
    def _generate_power_vec_data(self):
        power = [random.randint(1, 4)]
        data = []
        for power_ in power:
            data.extend(
                [dict(n_reps=n_reps, power=power_) for n_reps in self.N_VEC_REPS]
            )
        return self.generate_tests(data)

    def differential_power_vec_test_data(self):
        return self._generate_power_vec_data()

    def inverse_differential_power_vec_test_data(self):
        return self._generate_power_vec_data()

    def differential_log_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_differential_log_vec_test_data(self):
        return self.generate_vec_data()

    def differential_exp_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_differential_exp_vec_test_data(self):
        return self.generate_vec_data()

    def logm_vec_test_data(self):
        return self.generate_vec_data()

    def expm_after_logm_test_data(self):
        return self.generate_random_data()

    def logm_after_expm_test_data(self):
        return self.generate_random_data()

    def cholesky_factor_vec_test_data(self):
        return self.generate_vec_data()

    def cholesky_factor_belongs_to_positive_lower_triangular_matrices_test_data(self):
        return self.generate_random_data()

    def differential_cholesky_factor_vec_test_data(self):
        return self.generate_vec_data()

    def differential_cholesky_factor_belongs_to_positive_lower_triangular_matrices_test_data(
        self,
    ):
        return self.generate_random_data()


class SPDMatricesTestData(SPDMatricesMixinsTestData, OpenSetTestData):
    pass


class SPDMatrices2TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[3.0, -1.0], [-1.0, 3.0]]), expected=True),
            dict(point=gs.array([[1.0, 1.0], [2.0, 1.0]]), expected=False),
            dict(
                point=gs.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]]),
                expected=[True, False],
            ),
        ]

        return self.generate_tests(data)

    def projection_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[1.0, 0.0], [0.0, 1.0]]),
            ),
            dict(
                point=gs.array([[-1.0, 0.0], [0.0, -2.0]]),
                expected=gs.array([[gs.atol, 0.0], [0.0, gs.atol]]),
            ),
        ]
        return self.generate_tests(data)

    def logm_test_data(self):
        data = [
            dict(
                mat=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[0.0, 0.0], [0.0, 0.0]]),
            )
        ]
        return self.generate_tests(data)

    def cholesky_factor_test_data(self):
        data = [
            dict(
                mat=gs.array([[[1.0, 2.0], [2.0, 5.0]], [[1.0, 0.0], [0.0, 1.0]]]),
                expected=gs.array([[[1.0, 0.0], [2.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]),
            ),
        ]
        return self.generate_tests(data)

    def differential_cholesky_factor_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[1.0, 1.0], [1.0, 1.0]]),
                base_point=gs.array([[4.0, 2.0], [2.0, 5.0]]),
                expected=gs.array([[1 / 4, 0.0], [3 / 8, 1 / 16]]),
            )
        ]
        return self.generate_tests(data)


class SPDMatrices3TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=False,
            ),
        ]
        return self.generate_tests(data)

    def cholesky_factor_test_data(self):
        data = [
            dict(
                mat=gs.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
                expected=gs.array(
                    [
                        [SQRT_2, 0.0, 0.0],
                        [0.0, SQRT_2, 0.0],
                        [0.0, 0.0, SQRT_2],
                    ]
                ),
            ),
        ]
        return self.generate_tests(data)

    def differential_power_test_data(self):
        data = [
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
        return self.generate_tests(data)

    def inverse_differential_power_test_data(self):
        data = [
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
        return self.generate_tests(data)

    def differential_log_test_data(self):
        data = [
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
        return self.generate_tests(data)

    def inverse_differential_log_test_data(self):
        data = [
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

        return self.generate_tests(data)

    def differential_exp_test_data(self):
        data = [
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
        return self.generate_tests(data)

    def inverse_differential_exp_test_data(self):
        data = [
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
        return self.generate_tests(data)


class SPDAffineMetricPower1TestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False


class SPDAffineMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    skips = (
        "parallel_transport_ivp_norm",
        "parallel_transport_bvp_norm",
    )


class SPD2AffineMetricPower1TestData(TestData):
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[EXP_2, 0.0], [0.0, EXP_2]]),
            )
        ]
        return self.generate_tests(data)

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                base_point=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                expected=gs.array([[-2 * LN_2, 0.0], [0.0, -2 * LN_2]]),
            )
        ]
        return self.generate_tests(data)


class SPD3AffineMetricPower05TestData(TestData):
    def inner_product_test_data(self):
        data = [
            dict(
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
                expected=gs.array(713 / 144),
            )
        ]
        return self.generate_tests(data)


class SPDBuresWassersteinMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {
        "dist_point_to_itself_is_zero": {"atol": 1e-6},
    }

    skips = (
        "parallel_transport_bvp_transported_is_tangent",
        "parallel_transport_ivp_transported_is_tangent",
        "parallel_transport_bvp_vec",
        "parallel_transport_ivp_vec",
    )


class SPD2BuresWassersteinMetricTestData(TestData):
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[4.0, 0.0], [0.0, 4.0]]),
            )
        ]
        return self.generate_tests(data)

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[4.0, 0.0], [0.0, 4.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[2.0, 0.0], [0.0, 2.0]]),
            )
        ]
        return self.generate_tests(data)

    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                point_b=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                expected=gs.array(2 + 4 - (2 * 2 * SQRT_2)),
            )
        ]
        return self.generate_tests(data)


class SPD3BuresWassersteinMetricTestData(TestData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                tangent_vec_b=gs.array(
                    [[1.0, 2.0, 4.0], [2.0, 3.0, 8.0], [4.0, 8.0, 5.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.5, 0.5], [0.0, 0.5, 1.5]]
                ),
                expected=gs.array(4.0),
            )
        ]
        return self.generate_tests(data)


class SPDEuclideanMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def exp_domain_vec_test_data(self):
        return self.generate_vec_data()


class SPD2EuclideanMetricPower1TestData(TestData):
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[3.0, 0.0], [0.0, 3.0]]),
            )
        ]
        return self.generate_tests(data)

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[1.0, 0.0], [0.0, 1.0]]),
            )
        ]
        return self.generate_tests(data)

    def parallel_transport_test_data(self):
        smoke_data = [
            dict(
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                direction=gs.array([[1.0, 0.0], [0.0, 0.5]]),
                expected=gs.array([[2.0, 0.0], [0.0, 2.0]]),
            )
        ]
        return self.generate_tests(smoke_data)


class SPD3EuclideanMetricPower1TestData(TestData):
    def exp_domain_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array(
                    [[-1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 1.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
                ),
                expected=gs.array([-3, 1]),
            )
        ]
        return self.generate_tests(data)


class SPD3EuclideanMetricPower05TestData(TestData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                tangent_vec_b=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.5, 1.5], [0.0, 1.5, 2.5]]
                ),
                expected=gs.array(3472 / 576),
            )
        ]
        return self.generate_tests(data)


class SPDLogEuclideanMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False


class SPD2LogEuclideanMetricTestData(TestData):
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[EXP_2, 0.0], [0.0, EXP_2]]),
            )
        ]
        return self.generate_tests(data)

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[LN_2, 0.0], [0.0, LN_2]]),
            )
        ]
        return self.generate_tests(data)

    def dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                point_b=gs.array([[EXP_1, 0.0], [0.0, EXP_1]]),
                expected=gs.array(SQRT_2),
            )
        ]
        return self.generate_tests(data)


class SPD3LogEuclideanMetricTestData(TestData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array(
                    [[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]]
                ),
                tangent_vec_b=gs.array(
                    [[1.0, 1.0, 3.0], [1.0, 1.0, 3.0], [3.0, 3.0, 4.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]
                ),
                expected=gs.array(5.0 + (4.0 * ((2 * LN_2) ** 2))),
            )
        ]
        return self.generate_tests(data)
