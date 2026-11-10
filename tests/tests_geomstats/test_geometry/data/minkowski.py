import math

from polpo.testing.data import CaseData

import geomstats.backend as gs

from .euclidean import EuclideanMetricTestData


class MinkowskiMetricTestData(EuclideanMetricTestData):
    def __init__(self, space=None, **kwargs):
        super().__init__(space, **kwargs)

        # the Minkowski metric is not positive definite
        xfails = set(self.method_marks.get("xfail", ())) | {
            "dist",
            "dist_is_log_norm",
            "dist_is_positive",
            "dist_is_symmetric",
            "dist_triangle_inequality",
            "norm",
            "norm_is_positive",
            "normalize",
            "normalize_is_unitary",
            "parallel_transport_bvp_norm",
            "parallel_transport_ivp_norm",
            "squared_dist_is_positive",
        }
        self.method_marks = {**self.method_marks, "xfail": xfails}


class Minkowski2MetricTestData(CaseData):
    def metric_matrix_test_data(self):
        data = [dict(base_point=None, expected=gs.array([[-1.0, 0.0], [0.0, 1.0]]))]
        return data

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
        return data

    def squared_norm_test_data(self):
        data = [
            dict(
                vector=gs.array([-2.0, 4.0]),
                base_point=None,
                expected=gs.array(12.0),
            )
        ]
        return data

    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([2.0, -math.sqrt(3)]),
                point_b=gs.array([4.0, math.sqrt(15)]),
                expected=gs.array(27.416407),
            )
        ]
        return data

    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([2.0, math.sqrt(3)]),
                base_point=gs.array([1.0, 0.0]),
                expected=gs.array([3.0, math.sqrt(3)]),
            )
        ]
        return data

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([2.0, math.sqrt(3)]),
                base_point=gs.array([-1.0, 0.0]),
                expected=gs.array([3.0, math.sqrt(3)]),
            )
        ]
        return data
