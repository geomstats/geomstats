import math

import geomstats.backend as gs
from geomstats.test.data import TestData

from .euclidean import EuclideanMetricTestData


class MinkowskiTestData(TestData):
    def basis_cardinality_test_data(self):
        return None


class MinkowskiMetricTestData(EuclideanMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    xfails = (
        "dist_is_positive",
        "dist_is_symmetric",
        "dist_is_log_norm",
        "dist_triangle_inequality",
        "squared_dist_is_positive",
        "metric_matrix_is_spd",
        "norm_is_positive",
        "normalize_vec",
        "normalize_is_unitary",
        "norm_vec",
        "dist_vec",
        "parallel_transport_bvp_norm",
        "parallel_transport_ivp_norm",
    )


class Minkowski2MetricTestData(TestData):
    def metric_matrix_test_data(self):
        data = [dict(base_point=None, expected=gs.array([[-1.0, 0.0], [0.0, 1.0]]))]
        return self.generate_tests(data)

    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([0.0, 1.0]),
                tangent_vec_b=gs.array([2.0, 10.0]),
                base_point=None,
                expected=gs.array(10.0),
            ),
            dict(
                tangent_vec_a=gs.array([[-1.0, 0.0], [1.0, 0.0], [2.0, math.sqrt(3)]]),
                tangent_vec_b=gs.array(
                    [
                        [2.0, -math.sqrt(3)],
                        [4.0, math.sqrt(15)],
                        [-4.0, math.sqrt(15)],
                    ]
                ),
                base_point=None,
                expected=gs.array([2.0, -4.0, 14.70820393]),
            ),
        ]
        return self.generate_tests(data)

    def squared_norm_test_data(self):
        data = [
            dict(
                vector=gs.array([-2.0, 4.0]),
                base_point=None,
                expected=gs.array(12.0),
            )
        ]
        return self.generate_tests(data)

    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([2.0, -math.sqrt(3)]),
                point_b=gs.array([4.0, math.sqrt(15)]),
                expected=gs.array(27.416407),
            )
        ]
        return self.generate_tests(data)

    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([2.0, math.sqrt(3)]),
                base_point=gs.array([1.0, 0.0]),
                expected=gs.array([3.0, math.sqrt(3)]),
            )
        ]
        return self.generate_tests(data)

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([2.0, math.sqrt(3)]),
                base_point=gs.array([-1.0, 0.0]),
                expected=gs.array([3.0, math.sqrt(3)]),
            )
        ]
        return self.generate_tests(data)
