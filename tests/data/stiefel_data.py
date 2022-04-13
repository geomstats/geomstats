import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.stiefel import Stiefel
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

    def random_point_belongs_test_data(self):
        smoke_space_args_list = [(2, 2), (3, 3), (4, 3), (3, 2)]
        smoke_n_points_list = [1, 2, 1, 2]

        belongs_atol = gs.atol * 100000
        return self._random_point_belongs_test_data(
            smoke_space_args_list,
            smoke_n_points_list,
            self.space_args_list,
            self.n_points_list,
            belongs_atol,
        )

    def to_tangent_is_tangent_test_data(self):
        is_tangent_atol = gs.atol * 1000

        return self._to_tangent_is_tangent_test_data(
            Stiefel,
            self.space_args_list,
            self.shape_list,
            self.n_vecs_list,
            is_tangent_atol,
        )

    def projection_belongs_test_data(self):
        return self._projection_belongs_test_data(
            self.space_args_list, self.shape_list, self.n_points_list, gs.atol * 100
        )

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

    def random_tangent_vec_is_tangent_test_data(self):
        return self._random_tangent_vec_is_tangent_test_data(
            Stiefel,
            self.space_args_list,
            self.n_vecs_list,
            is_tangent_atol=gs.atol * 1000,
        )


class StiefelCanonicalMetricTestData(_RiemannianMetricTestData):

    n_list = random.sample(range(3, 5), 2)
    p_list = [random.sample(range(2, n), 1)[0] for n in n_list]
    metric_args_list = list(zip(n_list, p_list))
    shape_list = metric_args_list
    space_list = [Stiefel(n, p) for n, p in metric_args_list]
    n_points_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    def log_two_sheets_error_test_data(self):
        stiefel = Stiefel(n=3, p=3)
        base_point = stiefel.random_point()
        det_base = gs.linalg.det(base_point)
        point = stiefel.random_point()
        det_point = gs.linalg.det(point)
        if gs.all(det_base * det_point > 0.0):
            point *= -1.0

        random_data = [
            dict(
                n=3,
                p=3,
                point=point,
                base_point=base_point,
                expected=pytest.raises(ValueError),
            )
        ]
        return self.generate_tests([], random_data)

    def exp_shape_test_data(self):
        return self._exp_shape_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
        )

    def log_shape_test_data(self):
        return self._log_shape_test_data(
            self.metric_args_list,
            self.space_list,
        )

    def squared_dist_is_symmetric_test_data(self):
        return self._squared_dist_is_symmetric_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_a_list,
            self.n_points_b_list,
            atol=gs.atol * 1000,
        )

    def exp_belongs_test_data(self):
        return self._exp_belongs_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            belongs_atol=gs.atol * 10000,
        )

    def log_is_tangent_test_data(self):
        return self._log_is_tangent_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            is_tangent_atol=gs.atol * 1000,
        )

    def geodesic_ivp_belongs_test_data(self):
        return self._geodesic_ivp_belongs_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_points_list,
            belongs_atol=gs.atol * 1000,
        )

    def geodesic_bvp_belongs_test_data(self):
        return self._geodesic_bvp_belongs_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            belongs_atol=gs.atol * 1000,
        )

    def exp_after_log_test_data(self):
        return self._exp_after_log_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            rtol=gs.rtol * 100,
            atol=gs.atol * 10000,
        )

    def log_after_exp_test_data(self):
        return self._log_after_exp_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            rtol=gs.rtol * 100,
            atol=gs.atol * 10000,
        )

    def exp_ladder_parallel_transport_test_data(self):
        return self._exp_ladder_parallel_transport_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            self.n_rungs_list,
            self.alpha_list,
            self.scheme_list,
            atol=1e-1,
        )

    def exp_geodesic_ivp_test_data(self):
        return self._exp_geodesic_ivp_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            self.n_points_list,
            rtol=gs.rtol * 1000,
            atol=gs.atol * 1000,
        )

    def dist_is_symmetric_test_data(self):
        return self._dist_is_symmetric_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_a_list,
            self.n_points_b_list,
            atol=gs.atol * 1000,
        )

    def dist_is_positive_test_data(self):
        return self._dist_is_positive_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_a_list,
            self.n_points_b_list,
        )

    def squared_dist_is_positive_test_data(self):
        return self._squared_dist_is_positive_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_a_list,
            self.n_points_b_list,
        )

    def dist_is_norm_of_log_test_data(self):
        return self._dist_is_norm_of_log_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_a_list,
            self.n_points_b_list,
        )

    def dist_point_to_itself_is_zero_test_data(self):
        return self._dist_point_to_itself_is_zero_test_data(
            self.metric_args_list, self.space_list, self.n_points_list
        )

    def inner_product_is_symmetric_test_data(self):
        return self._inner_product_is_symmetric_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        )

    def triangle_inequality_of_dist_test_data(self):
        return self._triangle_inequality_of_dist_test_data(
            self.metric_args_list, self.space_list, self.n_points_list
        )

    def retraction_lifting_test_data(self):
        return self._log_after_exp_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            rtol=gs.rtol * 100,
            atol=gs.atol * 10000,
        )

    def lifting_retraction_test_data(self):
        return self._exp_after_log_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            rtol=gs.rtol * 100,
            atol=gs.atol * 10000,
        )

    def retraction_shape_test_data(self):
        return self.exp_shape_test_data()

    def lifting_shape_test_data(self):
        return self.log_shape_test_data()
