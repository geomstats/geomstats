from polpo.testing.data import CaseData

import geomstats.backend as gs


class OpenSimplex3TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([0.1, 0.2, 0.3, 0.4]), expected=gs.array(True)),
            dict(point=gs.array([0.0, 1.0, 0.3, 0.4]), expected=gs.array(False)),
            dict(point=gs.array([-0.5, 1.0, 0.3, 0.2]), expected=gs.array(False)),
            dict(
                point=gs.array([[0.1, 0.2, 0.3, 0.4], [-0.5, 1.0, 0.3, 0.2]]),
                expected=gs.array([True, False]),
            ),
        ]
        return data
