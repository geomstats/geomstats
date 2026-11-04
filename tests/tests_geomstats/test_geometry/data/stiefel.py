from polpo.testing.data import CaseData
from polpo.testing.lazy import LazyValue

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.random import get_data_generator

from .base import LevelSetTestData
from .riemannian_metric import RiemannianMetricTestData


class StiefelTestData(LevelSetTestData):
    def to_grassmannian_vec_test_data(self):
        return self.generate_vectorization_data(arg_names="point")

    def to_grassmannian_belongs_to_grassmannian_test_data(self):
        return self.generate_random_data(arg_names="point")


class StiefelStaticMethodsTestData(CaseData):
    def to_grassmannian_test_data(self):
        p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        r_z = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        point1 = gs.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]) / gs.sqrt(2.0)
        batch_points = Matrices.mul(
            GeneralLinear.exp(gs.stack([gs.pi * r_z / n for n in [2, 3, 4]])),
            point1,
        )
        return [
            dict(point=point1, expected=p_xy),
            dict(point=batch_points, expected=gs.stack([p_xy, p_xy, p_xy])),
        ]


class StiefelCanonicalMetricTestData(RiemannianMetricTestData):
    def log_antipodal_raises_test_data(self):
        return self.generate_random_data(arg_names="point")


class StiefelCanonicalMetric42TestData(CaseData):
    def log_partially_antipodal_raises_test_data(self):
        base_point = gs.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        point = gs.array([[-1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        return [dict(point=point, base_point=base_point)]


class SquareStiefelCanonicalMetricTestData(RiemannianMetricTestData):
    def __init__(self, space=None, **kwargs):
        super().__init__(space, **kwargs)

        # sample on SO(n) to keep random points on the same sheet
        self.data_generator = LazyValue(
            lambda: get_data_generator(
                SpecialOrthogonal(self.space.n, equip=False), **self.generator_kwargs
            )
        )

    def two_sheets_error_test_data(self):
        return self.generate_random_data(arg_names=("point", "base_point"))
