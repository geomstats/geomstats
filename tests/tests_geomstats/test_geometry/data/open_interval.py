from polpo.testing.data import CaseData

import geomstats.backend as gs


class OpenInterval01TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([0.5]), expected=gs.array(True)),
            dict(point=gs.array([1.5]), expected=gs.array(False)),
            dict(
                point=gs.array([[-0.5], [0.5], [1.5]]),
                expected=gs.array([False, True, False]),
            ),
        ]
        return data

    def projection_test_data(self):
        data = [
            dict(point=gs.array([0.5]), expected=gs.array([0.5])),
            dict(
                point=gs.array([[-0.5], [1.5]]),
                expected=gs.array([[gs.atol], [1 - gs.atol]]),
            ),
        ]
        return data


class OpenInterval01MetricTestData(CaseData):
    def injectivity_radius_test_data(self):
        data = [
            dict(base_point=gs.array([0.2]), expected=gs.array(0.2)),
            dict(
                base_point=gs.array([[0.2], [0.5], [0.9]]),
                expected=gs.array([0.2, 0.5, 0.1]),
            ),
        ]
        return data
