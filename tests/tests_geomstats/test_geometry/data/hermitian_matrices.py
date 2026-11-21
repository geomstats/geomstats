from polpo.testing.data import CaseData

import geomstats.backend as gs


class HermitianMatrices2TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[1.0, 2.0 + 1j], [2.0 - 1j, 1.0]]), expected=True),
            dict(point=gs.array([[1.0, 1.0], [2.0, 1.0]]), expected=False),
            dict(
                point=gs.array(
                    [[[1.0, 1j], [-1j, 1.0]], [[1.0 + 1j, -1.0], [0.0, 1.0]]]
                ),
                expected=[True, False],
            ),
        ]
        return data
