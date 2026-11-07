from polpo.testing.data import CaseData

import geomstats.backend as gs


class FullRankMatrices32TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.array(
                    [
                        [-1.6473486, -1.18240309],
                        [0.1944016, 0.18169231],
                        [-1.13933855, -0.64971248],
                    ]
                ),
                expected=True,
            ),
            dict(
                point=gs.array([[1.0, -1.0], [1.0, -1.0], [0.0, 0.0]]),
                expected=False,
            ),
        ]
        return data
