from polpo.testing.data import CaseData

import geomstats.backend as gs

from ...test_geometry.data.base import VectorSpaceOpenSetTestData
from .base import InformationManifoldMixinTestData


class BinomialDistributionsTestData(
    InformationManifoldMixinTestData, VectorSpaceOpenSetTestData
):
    def point_to_pdf_against_scipy_test_data(self):
        return self._generate_random_data_with_samples()


class Binomial5MetricTestData(CaseData):
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
        return data

    def metric_matrix_test_data(self):
        data = [
            dict(
                base_point=gs.array([0.5]),
                expected=gs.array([[20.0]]),
            ),
        ]
        return data


class Binomial7MetricTestData(CaseData):
    def metric_matrix_test_data(self):
        data = [
            dict(
                base_point=gs.array([[0.1], [0.5], [0.4]]),
                expected=gs.array(
                    [[[77.77777777777777]], [[28.0]], [[29.166666666666668]]]
                ),
            ),
        ]
        return data


class Binomial10MetricTestData(CaseData):
    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([0.1]),
                point_b=gs.array([0.99]),
                expected=gs.array(52.79685863761384),
            ),
        ]
        return data
