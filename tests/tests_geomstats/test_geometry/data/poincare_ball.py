import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import OpenSetTestData
from .riemannian_metric import RiemannianMetricTestData


class PoincareBallTestData(OpenSetTestData):
    xfails = ("projection_belongs",)


class PoincareBallMetricTestData(RiemannianMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False

    def mobius_add_vec_test_data(self):
        return self.generate_vec_data()

    def retraction_vec_test_data(self):
        return self.generate_vec_data()


class PoincareBall2TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([0.3, 0.5]), expected=True),
            dict(point=gs.array([1.2, 0.5]), expected=False),
        ]
        return self.generate_tests(data)

    def projection_norm_less_than_1_test_data(self):
        data = [dict(point=gs.array([1.2, 0.5]))]
        return self.generate_tests(data)


class PoincareBall2MetricTestData(TestData):
    def log_test_data(self):
        data = [
            dict(
                point=gs.array([0.3, 0.5]),
                base_point=gs.array([0.3, 0.3]),
                expected=gs.array([-0.01733576, 0.21958634]),
            )
        ]
        return self.generate_tests(data)

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
        return self.generate_tests(data)
