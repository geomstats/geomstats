import math

import geomstats.backend as gs
from geomstats.test.data import TestData

from .complex_riemannian_metric import ComplexRiemannianMetricTestData

LN_3 = math.log(3.0)
EXP_4 = math.exp(4.0)


class Siegel2TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[0.2 + 0.2j, -0.1j], [-0.3j, -0.1]]), expected=True),
            dict(point=gs.array([[1.0, 1.0], [2.0, 1.0]]), expected=False),
            dict(
                point=gs.array(
                    [
                        [[0.1 + 0.2j, 0.1 + 0.2j], [0.0 - 0.2j, 0.5j]],
                        [[1.0 - 1j, -1.0 + 0.2j], [0.4j, 1.0 + 1.2j]],
                    ]
                ),
                expected=[True, False],
            ),
        ]
        return self.generate_tests(data)

    def projection_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[1.0 - gs.atol, 0.0], [0.0, 1.0 - gs.atol]]),
            ),
            dict(
                point=gs.array([[1j, 0.0], [0.0, 1j]]),
                expected=gs.array(
                    [[(1.0 - gs.atol) * 1j, 0.0], [0.0, (1.0 - gs.atol) * 1j]]
                ),
            ),
            dict(
                point=gs.array([[-1.0, 0.0], [0.0, -2.0]]),
                expected=gs.array([[-(1 - gs.atol) / 2, 0.0], [0.0, -(1 - gs.atol)]]),
            ),
            dict(
                point=gs.array([[-1j, 0.0], [0.0, -2j]]),
                expected=gs.array(
                    [[-(1 - gs.atol) / 2 * 1j, 0.0], [0.0, -(1 - gs.atol) * 1j]]
                ),
            ),
        ]
        return self.generate_tests(data)


class SiegelMetricTestData(ComplexRiemannianMetricTestData):
    fail_for_not_implemented_errors = False

    def tangent_vec_from_base_point_to_zero_vec_test_data(self):
        return self.generate_vec_data()

    def tangent_vec_from_base_point_to_zero_is_tangent_test_data(self):
        return self.generate_random_data()

    # def sectional_curvature_at_zero_vec_test_data(self):
    #     return self.generate_vec_data()


class Siegel2MetricTestData(TestData):
    def exp_test_data(self):
        smoke_data = [
            dict(
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[0.0, 0.0], [0.0, 0.0]]),
                expected=gs.array(
                    [
                        [(EXP_4 - 1) / (EXP_4 + 1), 0.0],
                        [0.0, (EXP_4 - 1) / (EXP_4 + 1)],
                    ]
                ),
            ),
            dict(
                tangent_vec=gs.array([[2j, 0j], [0j, 2j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=gs.array(
                    [
                        [(EXP_4 - 1) / (EXP_4 + 1) * 1j, 0j],
                        [0j, (EXP_4 - 1) / (EXP_4 + 1) * 1j],
                    ]
                ),
            ),
            dict(
                tangent_vec=gs.array([[0j, 0j], [0j, 0j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=gs.array([[0j, 0j], [0j, 0j]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                point=gs.array([[0.5, 0.0], [0.0, 0.5]]),
                base_point=gs.array([[0.0, 0.0], [0.0, 0.0]]),
                expected=gs.array([[LN_3 / 2, 0.0], [0.0, LN_3 / 2]]),
            ),
            dict(
                point=gs.array([[0.5j, 0j], [0j, 0.5j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=gs.array([[LN_3 / 2 * 1j, 0j], [0j, LN_3 / 2 * 1j]]),
            ),
            dict(
                point=gs.array([[0j, 0j], [0j, 0j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=gs.array([[0j, 0j], [0j, 0j]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def sectional_curvature_test_data(self):
        smoke_data = [
            dict(
                tangent_vec_a=gs.array([[1.0, 0.0], [0.0, 0.0]]),
                tangent_vec_b=gs.array([[0.0, 0.0], [0.0, 1.0]]),
                base_point=gs.array([[0.0, 0.0], [0.0, 0.0]]),
                expected=gs.array(0.0),
            ),
            dict(
                tangent_vec_a=gs.array([[1j, 0j], [0j, 1j]]),
                tangent_vec_b=gs.array([[-1j, 0j], [0j, -1j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=gs.array(0.0),
            ),
            dict(
                tangent_vec_a=gs.array([[1.0 + 0j, 0j], [0j, 0j]]),
                tangent_vec_b=gs.array([[0j, 1j], [0j, 0j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=gs.array(-1.0),
            ),
            dict(
                tangent_vec_a=gs.array([[2 + 0j, 0j], [0j, 0j]]),
                tangent_vec_b=gs.array([[0j, 2j], [0j, 0j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=gs.array(-1.0),
            ),
            dict(
                tangent_vec_a=gs.array([[0.25 + 0j, 0j], [0j, 0j]]),
                tangent_vec_b=gs.array([[0j, 0.25j], [0j, 0j]]),
                base_point=gs.array([[0j, 0j], [0j, 0j]]),
                expected=gs.array(-1.0),
            ),
        ]
        return self.generate_tests(smoke_data)


class Siegel3MetricTestData(TestData):
    def inner_product_test_data(self):
        smoke_data = [
            dict(
                tangent_vec_a=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
                tangent_vec_b=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
                ),
                base_point=gs.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                expected=gs.array(6.0),
            ),
            dict(
                tangent_vec_a=gs.array([[1j, 0j, 0j], [0j, 1j, 0j], [0j, 0j, 1j]]),
                tangent_vec_b=gs.array([[1j, 0j, 0j], [0j, 2j, 0j], [0j, 0j, 3j]]),
                base_point=gs.array([[0j, 0j, 0j], [0j, 0j, 0j], [0j, 0j, 0j]]),
                expected=gs.array(6.0),
            ),
        ]
        return self.generate_tests(smoke_data)

    def sectional_curvature_test_data(self):
        smoke_data = [
            dict(
                tangent_vec_a=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
                tangent_vec_b=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
                ),
                base_point=gs.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                expected=gs.array(0.0),
            ),
            dict(
                tangent_vec_a=gs.array([[1j, 0j, 0j], [0j, 1j, 0j], [0j, 0j, 1j]]),
                tangent_vec_b=gs.array([[1j, 0j, 0j], [0j, 2j, 0j], [0j, 0j, 3j]]),
                base_point=gs.array([[0j, 0j, 0j], [0j, 0j, 0j], [0j, 0j, 0j]]),
                expected=gs.array(0.0),
            ),
            dict(
                tangent_vec_a=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                tangent_vec_b=gs.array(
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                base_point=gs.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                expected=gs.array(-1.0),
            ),
            dict(
                tangent_vec_a=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                tangent_vec_b=gs.array(
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                base_point=gs.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                expected=gs.array(-1.0),
            ),
            dict(
                tangent_vec_a=gs.array([[1j, 0j, 0j], [0j, 0j, 0j], [0j, 0j, 0j]]),
                tangent_vec_b=gs.array([[0j, 0j, 1j], [0j, 0j, 0j], [0j, 0j, 0j]]),
                base_point=gs.array([[0j, 0j, 0j], [0j, 0j, 0j], [0j, 0j, 0j]]),
                expected=gs.array(-1.0),
            ),
        ]
        return self.generate_tests(smoke_data)
