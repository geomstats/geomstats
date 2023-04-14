import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.test.data import TestData
from tests2.data.base_data import LevelSetTestData, RiemannianMetricTestData


class StiefelTestData(LevelSetTestData):
    def to_grassmannian_vec_test_data(self):
        return self.generate_vec_data()


class StiefelStaticMethodsTestData(TestData):
    def to_grassmannian_test_data(self):
        p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        r_z = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        point1 = gs.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]) / gs.sqrt(2.0)
        batch_points = Matrices.mul(
            GeneralLinear.exp(gs.array([gs.pi * r_z / n for n in [2, 3, 4]])),
            point1,
        )
        data = [
            dict(point=point1, expected=p_xy),
            dict(point=batch_points, expected=gs.array([p_xy, p_xy, p_xy])),
        ]
        return self.generate_tests(data)


class StiefelCanonicalMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
