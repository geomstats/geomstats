import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.stiefel import Stiefel, StiefelCanonicalMetric
from tests.data_generation import _LevelSetTestData, _RiemannianMetricTestData

p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
r_z = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
point1 = gs.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

point_a = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

point_b = gs.array(
    [
        [1.0 / gs.sqrt(2.0), 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0 / gs.sqrt(2.0), 0.0, 0.0],
    ]
)


class StiefelTestData(_LevelSetTestData):
    n_list = random.sample(range(2, 4), 2)
    p_list = [random.sample(range(2, n + 1), 1)[0] for n in n_list]
    space_args_list = list(zip(n_list, p_list))
    shape_list = space_args_list
    n_points_list = random.sample(range(1, 5), 2)
    n_vecs_list = random.sample(range(1, 5), 2)

    Space = Stiefel

    tolerances = {
        "random_tangent_vec_is_tangent": {"atol": 1e-8},
    }

    def to_grassmannian_test_data(self):

        point1 = gs.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]) / gs.sqrt(2.0)
        batch_points = Matrices.mul(
            GeneralLinear.exp(gs.array([gs.pi * r_z / n for n in [2, 3, 4]])),
            point1,
        )
        smoke_data = [
            dict(point=point1, expected=p_xy),
            dict(point=batch_points, expected=gs.array([p_xy, p_xy, p_xy])),
        ]
        return self.generate_tests(smoke_data)


class StiefelCanonicalMetricTestData(_RiemannianMetricTestData):

    n_list = random.sample(range(3, 5), 2)
    p_list = [random.sample(range(2, n), 1)[0] for n in n_list]

    shape_list = space_args_list = list(zip(n_list, p_list))
    space_list = [Stiefel(n, p) for n, p in space_args_list]
    metric_args_list = [{} for _ in shape_list]

    n_points_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    Metric = StiefelCanonicalMetric

    def log_two_sheets_error_test_data(self):
        space = Stiefel(n=3, p=3, equip=False)
        base_point = space.random_point()
        det_base = gs.linalg.det(base_point)
        point = space.random_point()
        det_point = gs.linalg.det(point)
        if gs.all(det_base * det_point > 0.0):
            point *= -1.0

        random_data = [
            dict(
                space=space,
                point=point,
                base_point=base_point,
                expected=pytest.raises(ValueError),
            )
        ]
        return self.generate_tests([], random_data)

    def retraction_lifting_test_data(self):
        return super().log_after_exp_test_data()

    def lifting_retraction_test_data(self):
        return super().exp_after_log_test_data()

    def retraction_shape_test_data(self):
        return self.exp_shape_test_data()

    def lifting_shape_test_data(self):
        return self.log_shape_test_data()
