import math

from polpo.testing.data import CaseData

import geomstats.backend as gs


class PositiveHypersphere1TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([0.6, 0.8]), expected=gs.array(True)),
            dict(point=gs.array([-0.6, 0.8]), expected=gs.array(False)),
            dict(point=gs.array([0.5, 0.5]), expected=gs.array(False)),
            dict(
                point=gs.array([[0.6, 0.8], [-0.6, 0.8]]),
                expected=gs.array([True, False]),
            ),
        ]
        return data

    def projection_test_data(self):
        data = [
            dict(point=gs.array([0.6, 0.8]), expected=gs.array([0.6, 0.8])),
            dict(point=gs.array([-1.0, 2.0]), expected=gs.array([0.0, 1.0])),
        ]
        return data


class PositiveHypersphere1MetricTestData(CaseData):
    def injectivity_radius_test_data(self):
        data = [
            dict(base_point=gs.array([0.6, 0.8]), expected=gs.array(math.asin(0.6))),
            dict(
                base_point=gs.array([[0.6, 0.8], [0.8, 0.6]]),
                expected=gs.array([math.asin(0.6), math.asin(0.6)]),
            ),
        ]
        return data
