import random

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.test.data import TestData

from .base import LevelSetTestData
from .riemannian_metric import RiemannianMetricTestData


class StiefelTestData(LevelSetTestData):
    def to_grassmannian_vec_test_data(self):
        return self.generate_vec_data()

    def to_grassmannian_belongs_to_grassmannian_test_data(self):
        return self.generate_random_data()


class StiefelStaticMethodsTestData(TestData):
    def to_grassmannian_test_data(self):
        p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        r_z = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        point1 = gs.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]) / gs.sqrt(2.0)
        batch_points = Matrices.mul(
            GeneralLinear.exp(gs.stack([gs.pi * r_z / n for n in [2, 3, 4]])),
            point1,
        )
        data = [
            dict(point=point1, expected=p_xy),
            dict(point=batch_points, expected=gs.stack([p_xy, p_xy, p_xy])),
        ]
        return self.generate_tests(data)


class StiefelCanonicalMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def retraction_vec_test_data(self):
        return self.generate_vec_data()

    def lifting_vec_test_data(self):
        return self.generate_vec_data()

    def lifting_is_tangent_test_data(self):
        return self.generate_random_data()

    def lifting_after_retraction_test_data(self):
        return self.generate_random_data()

    def retraction_after_lifting_test_data(self):
        return self.generate_random_data()


class StiefelCanonicalMetricSquareTestData(TestData):
    def two_sheets_error_test_data(self):
        return self.generate_random_data()


class StiefelConnectednessTestData(TestData):
    def is_connected_test_data(self):
        n = random.randint(2, 8)
        n_gt_3 = random.randint(3, 8)
        n_gt_5 = random.randint(5, 8)
        p = random.randint(2, n_gt_5 - 2)

        data = [
            dict(point=(n, 1), expected=True),
            dict(point=(n, n), expected=False),
            dict(point=(n_gt_3, n_gt_3 - 1), expected=False),
            dict(point=(n_gt_5, p), expected=True),
        ]
        return self.generate_tests(data)
