import math

import pytest

import geomstats.backend as gs

from .base import OpenSetTestData
from .riemannian_metric import RiemannianMetricTestData

LN_2 = math.log(2.0)
EXP_1 = math.exp(1.0)
EXP_2 = math.exp(2.0)


class PositiveRealsTestData(OpenSetTestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[10.0]]), expected=[True]),
            dict(point=gs.array([[10 + 0j]]), expected=[True]),
            dict(point=gs.array([[10 + 1j]]), expected=[False]),
            dict(point=gs.array([[-10.0]]), expected=[False]),
            dict(
                point=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=[False, False, False],
            ),
            dict(point=gs.array([[1.0], [-1.5]]), expected=[True, False]),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))

    def projection_test_data(self):
        data = [
            dict(point=gs.array([[1.0]]), expected=gs.array([[1.0]])),
            dict(
                point=gs.array([[-1.0]]),
                expected=gs.array([[gs.atol]]),
            ),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))


class PositiveRealsMetricTestData(RiemannianMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False

    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([1.0]),
                tangent_vec_b=gs.array([2.0]),
                base_point=gs.array([3.0]),
                expected=gs.array(2 / 9),
            ),
            dict(
                tangent_vec_a=gs.array([-2.0]),
                tangent_vec_b=gs.array([3.0]),
                base_point=gs.array([4.0]),
                expected=gs.array(-3 / 8),
            ),
        ]
        return self.generate_tests(data)

    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[1.0]]),
                base_point=gs.array([[1.0]]),
                expected=gs.array([[EXP_1]]),
            ),
            dict(
                tangent_vec=gs.array([[4.0]]),
                base_point=gs.array([[2.0]]),
                expected=gs.array([[2 * EXP_2]]),
            ),
            dict(
                tangent_vec=gs.array([[1.0], [2.0]]),
                base_point=gs.array([[1.0]]),
                expected=gs.array([[EXP_1], [EXP_2]]),
            ),
        ]
        return self.generate_tests(data)

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[4.0]]),
                base_point=gs.array([[2.0]]),
                expected=gs.array([[2 * LN_2]]),
            ),
            dict(
                point=gs.array([[2.0]]),
                base_point=gs.array([[4.0]]),
                expected=gs.array([[-4 * LN_2]]),
            ),
            dict(
                point=gs.array([[1.0], [2.0]]),
                base_point=gs.array([[1.0]]),
                expected=gs.array([[0], [LN_2]]),
            ),
        ]
        return self.generate_tests(data)
