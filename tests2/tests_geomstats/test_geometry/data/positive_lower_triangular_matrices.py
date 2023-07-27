import math

import geomstats.backend as gs
from geomstats.test.data import TestData
from tests2.tests_geomstats.test_geometry.data.invariant_metric import (
    InvariantMetricMatrixTestData,
)
from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricTestData,
)

from .base import OpenSetTestData

EULER = gs.exp(1.0)
SQRT_2 = math.sqrt(2)


class PositiveLowerTriangularMatricesTestData(OpenSetTestData):
    def gram_vec_test_data(self):
        return self.generate_vec_data()

    def differential_gram_vec_test_data(self):
        return self.generate_vec_data()

    def inverse_differential_gram_vec_test_data(self):
        return self.generate_vec_data()

    def gram_belongs_to_spd_matrices_test_data(self):
        return self.generate_random_data()

    def differential_gram_belongs_to_symmetric_matrices_test_data(self):
        return self.generate_random_data()

    def inverse_differential_gram_belongs_to_lower_triangular_matrices_test_data(self):
        return self.generate_random_data()


class PositiveLowerTriangularMatrices2TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[1.0, 0.0], [-1.0, 3.0]]), expected=True),
            dict(point=gs.array([[1.0, -1.0], [-1.0, 3.0]]), expected=False),
            dict(point=gs.array([[-1.0, 0.0], [-1.0, 3.0]]), expected=False),
            dict(
                point=gs.array(
                    [
                        [[1.0, 0], [0, 1.0]],
                        [[1.0, 2.0], [2.0, 1.0]],
                        [[-1.0, 0.0], [1.0, 1.0]],
                        [[0.0, 0.0], [1.0, 1.0]],
                    ]
                ),
                expected=gs.array([True, False, False, False]),
            ),
            dict(
                point=gs.array(
                    [
                        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[-1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    ]
                ),
                expected=gs.array([False, False, False, False]),
            ),
        ]
        return self.generate_tests(data)

    def gram_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, 0.0], [2.0, 1.0]]),
                expected=gs.array([[1.0, 2.0], [2.0, 5.0]]),
            ),
            dict(
                point=gs.array([[[2.0, 1.0], [0.0, 1.0]], [[-6.0, 0.0], [5.0, 3.0]]]),
                expected=gs.array(
                    [[[5.0, 1.0], [1.0, 1.0]], [[36.0, -30.0], [-30.0, 34.0]]]
                ),
            ),
        ]
        return self.generate_tests(data)

    def differential_gram_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[-1.0, 0.0], [2.0, -1.0]]),
                base_point=gs.array([[1.0, 0.0], [2.0, 1.0]]),
                expected=gs.array([[-2.0, 0.0], [0.0, 6.0]]),
            ),
            dict(
                tangent_vec=gs.array(
                    [[[-1.0, 2.0], [2.0, -1.0]], [[0.0, 4.0], [4.0, -1.0]]]
                ),
                base_point=gs.array(
                    [[[3.0, 0.0], [-1.0, 2.0]], [[4.0, 0.0], [-1.0, 4.0]]]
                ),
                expected=gs.array(
                    [
                        [[-6.0, 11.0], [11.0, -8.0]],
                        [[0.0, 32.0], [32.0, -16.0]],
                    ]
                ),
            ),
        ]
        return self.generate_tests(data)

    def inverse_differential_gram_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[1.0, 2.0], [2.0, 5.0]]),
                base_point=gs.array([[1.0, 0.0], [2.0, 2.0]]),
                expected=gs.array([[0.5, 0.0], [1.0, 0.25]]),
            ),
            dict(
                tangent_vec=gs.array(
                    [[[-4.0, 1.0], [1.0, -4.0]], [[0.0, 4.0], [4.0, -8.0]]]
                ),
                base_point=gs.array(
                    [[[2.0, 0.0], [-1.0, 2.0]], [[4.0, 0.0], [-1.0, 2.0]]]
                ),
                expected=gs.array(
                    [[[-1.0, 0.0], [0.0, -1.0]], [[0.0, 0.0], [1.0, -1.5]]]
                ),
            ),
        ]
        return self.generate_tests(data)


class CholeskyMetricTestData(RiemannianMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False

    def diag_inner_product_vec_test_data(self):
        return self.generate_vec_data()

    def strictly_lower_inner_product_vec_test_data(self):
        return self.generate_vec_data()


class CholeskyMetric2TestData(TestData):
    def diag_inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([[1.0, 0.0], [-2.0, -1.0]]),
                tangent_vec_b=gs.array([[2.0, 0.0], [-3.0, -1.0]]),
                base_point=gs.array([[SQRT_2, 0.0], [-3.0, 1.0]]),
                expected=gs.array(2.0),
            )
        ]
        return self.generate_tests(data)

    def strictly_lower_inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([[1.0, 0.0], [-2.0, -1.0]]),
                tangent_vec_b=gs.array([[2.0, 0.0], [-3.0, -1.0]]),
                expected=gs.array(6.0),
            )
        ]
        return self.generate_tests(data)

    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([[1.0, 0.0], [-2.0, -1.0]]),
                tangent_vec_b=gs.array([[2.0, 0.0], [-3.0, -1.0]]),
                base_point=gs.array([[SQRT_2, 0.0], [-3.0, 1.0]]),
                expected=gs.array(8.0),
            ),
            dict(
                tangent_vec_a=gs.array(
                    [
                        [[3.0, 0.0], [4.0, 2.0]],
                        [[-1.0, 0.0], [2.0, -4.0]],
                    ]
                ),
                tangent_vec_b=gs.array(
                    [[[4.0, 0.0], [3.0, 3.0]], [[3.0, 0.0], [-6.0, 2.0]]]
                ),
                base_point=gs.array([[[3, 0.0], [-2.0, 6.0]], [[1, 0.0], [-1.0, 1.0]]]),
                expected=gs.array([13.5, -23.0]),
            ),
        ]
        return self.generate_tests(data)

    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[-1.0, 0.0], [2.0, 3.0]]),
                base_point=gs.array([[1.0, 0.0], [2.0, 2.0]]),
                expected=gs.array([[1 / EULER, 0.0], [4.0, 2 * gs.exp(1.5)]]),
            ),
            dict(
                tangent_vec=gs.array(
                    [[[0.0, 0.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]
                ),
                base_point=gs.array(
                    [[[1.0, 0.0], [2.0, 2.0]], [[1.0, 0.0], [0.0, 2.0]]]
                ),
                expected=gs.array(
                    [
                        [[1.0, 0.0], [4.0, 2.0]],
                        [[gs.exp(1.0), 0.0], [0.0, 2.0]],
                    ]
                ),
            ),
        ]
        return self.generate_tests(data)

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[EULER, 0.0], [2.0, EULER**3]]),
                base_point=gs.array([[EULER**3, 0.0], [4.0, EULER**4]]),
                expected=gs.array([[-2.0 * EULER**3, 0.0], [-2.0, -1 * EULER**4]]),
            ),
            dict(
                point=gs.array(
                    [
                        [[gs.exp(-2.0), 0.0], [0.0, gs.exp(2.0)]],
                        [[gs.exp(-3.0), 0.0], [2.0, gs.exp(3.0)]],
                    ]
                ),
                base_point=gs.array(
                    [[[1.0, 0.0], [-1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
                ),
                expected=gs.array(
                    [[[-2.0, 0.0], [1.0, 2.0]], [[-3.0, 0.0], [2.0, 3.0]]]
                ),
            ),
        ]
        return self.generate_tests(data)

    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([[EULER, 0.0], [2.0, EULER**3]]),
                point_b=gs.array([[EULER**3, 0.0], [4.0, EULER**4]]),
                expected=gs.array(9.0),
            ),
            dict(
                point_a=gs.array(
                    [
                        [[EULER, 0.0], [2.0, EULER**3]],
                        [[EULER, 0.0], [4.0, EULER**3]],
                    ]
                ),
                point_b=gs.array(
                    [
                        [[EULER**3, 0.0], [4.0, EULER**4]],
                        [[EULER**3, 0.0], [7.0, EULER**4]],
                    ]
                ),
                expected=gs.array([9.0, 14.0]),
            ),
        ]
        return self.generate_tests(data)


class InvariantPositiveLowerTriangularMatricesMetricTestData(
    InvariantMetricMatrixTestData
):
    trials = 3
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False

    tolerances = {
        "exp_after_log_at_identity": {"atol": 1e-4},
        "geodesic_boundary_points": {"atol": 1e-4},
        "geodesic_bvp_reverse": {"atol": 1e-4},
        "log_after_exp": {"atol": 1e-4},
        "parallel_transport_bvp_norm": {"atol": 1e-4},
        "parallel_transport_ivp_norm": {"atol": 1e-4},
        "squared_dist_is_symmetric": {"atol": 1e-4},
    }
