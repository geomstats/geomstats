import math

import geomstats.backend as gs
from geomstats.test.data import TestData

from .complex_riemannian_metric import ComplexRiemannianMetricTestData

LN_3 = math.log(3.0)
EXP_4 = math.exp(4.0)
SQRT_2 = math.sqrt(2.0)
SQRT_8 = math.sqrt(8.0)


class ComplexPoincareDiskSmokeTestData(TestData):
    def belongs_test_data(self):
        data = [
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
        return self.generate_tests(data)

    def projection_test_data(self):
        data = [
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
        return self.generate_tests(data)


class ComplexPoincareDiskMetricTestData(ComplexRiemannianMetricTestData):
    fail_for_not_implemented_errors = False
    skips = ("sectional_curvature_vec",)


class ComplexPoincareDiskMetricSmokeTestData(TestData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([[1j]]),
                tangent_vec_b=gs.array([[1j]]),
                base_point=gs.array([[0j]]),
                expected=gs.array([1.0]),
            ),
            dict(
                tangent_vec_a=gs.array([[3.0 + 4j]]),
                tangent_vec_b=gs.array([[3.0 + 4j]]),
                base_point=gs.array([[0.0]]),
                expected=gs.array([25.0]),
            ),
            dict(
                tangent_vec_a=gs.array([[1j], [1j], [1j]]),
                tangent_vec_b=gs.array([[1j], [2j], [3j]]),
                base_point=gs.array([[0j], [0j], [0j]]),
                expected=gs.array([1.0, 2.0, 3.0]),
            ),
        ]
        return self.generate_tests(data)

    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[2.0 + 0j]]),
                base_point=gs.array([[0.0 + 0j]]),
                expected=gs.array([[(EXP_4 - 1) / (EXP_4 + 1)]]),
            ),
            dict(
                tangent_vec=gs.array([[2.0 + 0j]]),
                base_point=gs.array([[0.0 + 0j]]),
                expected=gs.array([[(EXP_4 - 1) / (EXP_4 + 1)]]),
            ),
            dict(
                tangent_vec=gs.array([[2.0 + 2j]]),
                base_point=gs.array([[0.0 + 0j]]),
                expected=gs.array(
                    [
                        [
                            (1 + 1j)
                            / SQRT_2
                            * (gs.exp(2 * SQRT_8 + 0j) - 1)
                            / (gs.exp(2 * SQRT_8 + 0j) + 1)
                        ]
                    ]
                ),
            ),
        ]
        return self.generate_tests(data)

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[0.5]]),
                base_point=gs.array([[0.0]]),
                expected=gs.array([[LN_3 / 2]]),
            )
        ]
        return self.generate_tests(data)
