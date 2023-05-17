import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import LevelSetTestData
from .matrices import MatricesMetricTestData


class GrassmannianTestData(LevelSetTestData):
    pass


class Grassmannian32TestData(TestData):
    def belongs_test_data(self):
        p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        p_yz = gs.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        p_xz = gs.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        data = [
            dict(point=p_xy, expected=gs.array(True)),
            dict(point=gs.stack([p_yz, p_xz]), expected=gs.array([True, True])),
        ]
        return self.generate_tests(data)


class GrassmannianCanonicalMetricTestData(MatricesMetricTestData):
    fail_for_not_implemented_errors = False
