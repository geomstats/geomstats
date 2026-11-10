from polpo.testing.data import CaseData

import geomstats.backend as gs

from .base import VectorSpaceOpenSetTestData
from .riemannian_metric import RiemannianMetricTestData


class PoincareBall2TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([0.3, 0.5]), expected=True),
            dict(point=gs.array([1.2, 0.5]), expected=False),
        ]
        return data

    def projection_norm_less_than_1_test_data(self):
        data = [dict(point=gs.array([1.2, 0.5]))]
        return data


class PoincareBallTestData(VectorSpaceOpenSetTestData):
    def __init__(self, space=None, **kwargs):
        super().__init__(space, **kwargs)

        xfails = set(self.method_marks.get("xfail", ())) | {"projection_belongs"}
        self.method_marks = {**self.method_marks, "xfail": xfails}


class PoincareBall2MetricTestData(CaseData):
    def log_test_data(self):
        data = [
            dict(
                point=gs.array([0.3, 0.5]),
                base_point=gs.array([0.3, 0.3]),
                expected=gs.array([-0.01733576, 0.21958634]),
            )
        ]
        return data

    def dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([0.5, 0.5]),
                point_b=gs.array([0.5, -0.5]),
                expected=gs.array(2.887270927429199),
            ),
            dict(
                point_a=gs.array([0.1, 0.2]),
                point_b=gs.array([[0.3, 0.4], [0.5, 0.5]]),
                expected=gs.array([0.65821943, 1.34682524]),
            ),
            dict(
                point_a=gs.array([0.3, 0.4]),
                point_b=gs.array([0.5, 0.5]),
                expected=gs.array(0.71497076),
            ),
        ]
        return data


class PoincareBallMetricTestData(RiemannianMetricTestData):
    def mobius_add_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point_a", "point_b"),
            on_metric=True,
        )
