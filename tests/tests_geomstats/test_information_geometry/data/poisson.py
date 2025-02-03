import geomstats.backend as gs
from geomstats.test.data import TestData

from ...test_geometry.data.base import VectorSpaceOpenSetTestData
from ...test_geometry.data.riemannian_metric import RiemannianMetricTestData
from .base import InformationManifoldMixinTestData


class PoissonDistributionsTestData(
    InformationManifoldMixinTestData, VectorSpaceOpenSetTestData
):
    fail_for_not_implemented_errors = False

    def point_to_pdf_against_scipy_test_data(self):
        return self.generate_random_data_with_samples()


class PoissonDistributionsSmokeTestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([5.0]), expected=True),
            dict(point=gs.array([-2.0]), expected=False),
        ]
        return self.generate_tests(data)


class PoissonMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False


class PoissonMetricSmokeTestData(TestData):
    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([1, 3, 0.1]),
                point_b=gs.array([4, 3, 0.9]),
                expected=gs.array([4.0, 0.0, 1.6]),
            ),
            dict(
                point_a=gs.array(0.1),
                point_b=gs.array(4.9),
                expected=gs.array(14.4),
            ),
            dict(
                point_a=gs.array(0.1),
                point_b=gs.array([4.9, 0.9]),
                expected=gs.array([14.4, 1.6]),
            ),
            dict(
                point_a=gs.array([4.9, 0.4]),
                point_b=gs.array(0.1),
                expected=gs.array([14.4, 0.4]),
            ),
        ]
        return self.generate_tests(data)

    def metric_matrix_test_data(self):
        data = [
            dict(
                base_point=gs.array([0.5]),
                expected=gs.array([[2.0]]),
            ),
            dict(
                base_point=gs.array([[0.5], [0.2]]),
                expected=gs.array([[[2.0]], [[5.0]]]),
            ),
        ]
        return self.generate_tests(data)
