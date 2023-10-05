import geomstats.backend as gs
from geomstats.test.data import TestData
from tests2.tests_geomstats.test_geometry.data.base import OpenSetTestData
from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricTestData,
)

from .base import InformationManifoldMixinTestData


class BinomialDistributionsTestData(InformationManifoldMixinTestData, OpenSetTestData):
    fail_for_not_implemented_errors = False

    def point_to_pdf_against_scipy_test_data(self):
        return self.generate_random_data_with_samples()


class BinomialMetricTestData(RiemannianMetricTestData):
    trials = 2
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    xfails = (
        # TODO: regularize tangent_vec
        "log_after_exp",
    )


class Binomial5MetricTestData(TestData):
    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([0.2, 0.3]),
                point_b=gs.array([0.3, 0.5]),
                expected=gs.array([0.26908349, 0.84673057]),
            ),
            dict(
                point_a=gs.array(0.3),
                point_b=gs.array([0.2, 0.5]),
                expected=gs.array([0.26908349, 0.84673057]),
            ),
            dict(
                point_a=gs.array([0.2, 0.5]),
                point_b=gs.array(0.3),
                expected=gs.array([0.26908349, 0.84673057]),
            ),
        ]
        return self.generate_tests(data)

    def metric_matrix_test_data(self):
        data = [
            dict(
                base_point=gs.array([0.5]),
                expected=gs.array([[20.0]]),
            ),
        ]
        return self.generate_tests(data)


class Binomial7MetricTestData(TestData):
    def metric_matrix_test_data(self):
        data = [
            dict(
                base_point=gs.array([[0.1], [0.5], [0.4]]),
                expected=gs.array(
                    [[[77.77777777777777]], [[28.0]], [[29.166666666666668]]]
                ),
            ),
        ]
        return self.generate_tests(data)


class Binomial10MetricTestData(TestData):
    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([0.1]),
                point_b=gs.array([0.99]),
                expected=gs.array(52.79685863761384),
            ),
        ]
        return self.generate_tests(data)
