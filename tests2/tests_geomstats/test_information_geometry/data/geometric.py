import geomstats.backend as gs
from geomstats.test.data import TestData

from ...test_geometry.data.base import OpenSetTestData
from ...test_geometry.data.riemannian_metric import RiemannianMetricTestData
from .base import InformationManifoldMixinTestData


class GeometricDistributionsTestData(InformationManifoldMixinTestData, OpenSetTestData):
    fail_for_not_implemented_errors = False

    def point_to_pdf_against_scipy_test_data(self):
        return self.generate_random_data_with_samples()


class GeometricDistributionsSmokeTestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([0.1]), expected=True),
            dict(point=gs.array([5.0]), expected=False),
            dict(point=gs.array([-2.0]), expected=False),
            dict(point=gs.array([[0.9], [-1.0]]), expected=gs.array([True, False])),
            dict(point=gs.array([[0.1], [0.7]]), expected=gs.array([True, True])),
            dict(point=gs.array([[-0.1], [3.7]]), expected=gs.array([False, False])),
        ]
        return self.generate_tests(data)


class GeometricMetricTestData(RiemannianMetricTestData):
    trials = 2
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False


class GeometricMetricSmokeTestData(TestData):
    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([0.2, 0.3]),
                point_b=gs.array([0.3, 0.5]),
                expected=gs.array([0.21846342154512002, 0.4318107273293949]),
            ),
            dict(
                point_a=gs.array(0.2),
                point_b=gs.array(0.3),
                expected=gs.array(0.21846342154512002),
            ),
            dict(
                point_a=gs.array(0.3),
                point_b=gs.array([0.2, 0.5]),
                expected=gs.array([0.21846342154512002, 0.4318107273293949]),
            ),
        ]
        return self.generate_tests(data)

    def metric_matrix_test_data(self):
        data = [
            dict(
                base_point=gs.array([0.5]),
                expected=gs.array([[8.0]]),
            ),
            dict(
                base_point=gs.array([[0.2], [0.4]]),
                expected=gs.array([[[31.249999999999993]], [[10.416666666666664]]]),
            ),
        ]
        return self.generate_tests(data)
