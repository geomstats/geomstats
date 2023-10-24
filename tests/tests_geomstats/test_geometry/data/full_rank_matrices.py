import geomstats.backend as gs
from geomstats.test.data import TestData


class FullRankMatrices32TestData(TestData):
    def belongs_test_data(self):
        smoke_data = [
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
        return self.generate_tests(smoke_data)
