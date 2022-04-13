import random

import geomstats.backend as gs
from geomstats.geometry.poincare_ball import PoincareBall
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData


class PoincareBallTestData(_OpenSetTestData):
    smoke_space_args_list = [(2,), (3,), (4,), (5,)]
    smoke_n_points_list = [1, 2, 1, 2]
    n_list = random.sample(range(2, 10), 5)
    space_args_list = [(n,) for n in n_list]
    n_points_list = random.sample(range(1, 10), 5)
    shape_list = [(n,) for n in n_list]
    n_vecs_list = random.sample(range(1, 10), 5)

    def belongs_test_data(self):
        smoke_data = [
            dict(dim=2, point=[0.3, 0.5], expected=True),
            dict(dim=2, point=[1.2, 0.5], expected=False),
        ]
        return self.generate_tests(smoke_data)

    def projection_norm_lessthan_1_test_data(self):
        smoke_data = [dict(dim=2, point=[1.2, 0.5])]
        return self.generate_tests(smoke_data)

    def random_point_belongs_test_data(self):
        belongs_atol = gs.atol * 100000
        return self._random_point_belongs_test_data(
            self.smoke_space_args_list,
            self.smoke_n_points_list,
            self.space_args_list,
            self.n_points_list,
            belongs_atol,
        )

    def to_tangent_is_tangent_test_data(self):

        is_tangent_atol = gs.atol * 1000

        return self._to_tangent_is_tangent_test_data(
            PoincareBall,
            self.space_args_list,
            self.shape_list,
            self.n_vecs_list,
            is_tangent_atol,
        )

    def projection_belongs_test_data(self):
        return self._projection_belongs_test_data(
            self.space_args_list, self.shape_list, self.n_points_list
        )

    def to_tangent_is_tangent_in_ambient_space_test_data(self):
        return self._to_tangent_is_tangent_in_ambient_space_test_data(
            PoincareBall, self.space_args_list, self.shape_list
        )

    def random_tangent_vec_is_tangent_test_data(self):
        return self._random_tangent_vec_is_tangent_test_data(
            PoincareBall, self.space_args_list, self.n_vecs_list
        )


class TestDataPoincareBallMetric(_RiemannianMetricTestData):
    n_list = random.sample(range(2, 5), 2)
    metric_args_list = [(n,) for n in n_list]
    shape_list = [(n,) for n in n_list]
    space_list = [PoincareBall(n) for n in n_list]
    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    def mobius_out_of_the_ball_test_data(self):
        smoke_data = [dict(dim=2, x=[0.7, 0.9], y=[0.2, 0.2])]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point=[0.3, 0.5],
                base_point=[0.3, 0.3],
                expected=[-0.01733576, 0.21958634],
            )
        ]
        return self.generate_tests(smoke_data)

    def dist_pairwise_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.5]],
                expected=[
                    [0.0, 0.65821943, 1.34682524],
                    [0.65821943, 0.0, 0.71497076],
                    [1.34682524, 0.71497076, 0.0],
                ],
            )
        ]
        return self.generate_tests(smoke_data)

    def dist_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point_a=[0.5, 0.5],
                point_b=[0.5, -0.5],
                expected=2.887270927429199,
            )
        ]
        return self.generate_tests(smoke_data)

    def coordinate_test_data(self):
        smoke_data = [dict(dim=2, point_a=[-0.3, 0.7], point_b=[0.2, 0.5])]
        return self.generate_tests(smoke_data)

    def exp_shape_test_data(self):
        return self._exp_shape_test_data(
            self.metric_args_list, self.space_list, self.shape_list
        )

    def log_shape_test_data(self):
        return self._log_shape_test_data(self.metric_args_list, self.space_list)

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
            belongs_atol=gs.atol * 100000,
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
            belongs_atol=gs.atol * 10000,
        )

    def geodesic_bvp_belongs_test_data(self):
        return self._geodesic_bvp_belongs_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            belongs_atol=gs.atol * 10000,
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
        )

    def exp_geodesic_ivp_test_data(self):
        return self._exp_geodesic_ivp_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            self.n_points_list,
            rtol=gs.rtol * 10000,
            atol=gs.atol * 10000,
        )

    def parallel_transport_ivp_is_isometry_test_data(self):
        return self._parallel_transport_ivp_is_isometry_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            is_tangent_atol=gs.atol * 1000,
            atol=gs.atol * 1000,
        )

    def parallel_transport_bvp_is_isometry_test_data(self):
        return self._parallel_transport_bvp_is_isometry_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            is_tangent_atol=gs.atol * 1000,
            atol=gs.atol * 1000,
        )

    def dist_is_symmetric_test_data(self):
        return self._dist_is_symmetric_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_a_list,
            self.n_points_b_list,
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

    def triangular_inequality_of_dist_test_data(self):
        return self._triangular_inequality_of_dist_test_data(
            self.metric_args_list, self.space_list, self.n_points_list
        )

    def mobius_vectorization_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point_a=gs.array([0.5, 0.5]),
                point_b=gs.array([[0.5, -0.3], [0.3, 0.4]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def log_vectorization_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point_a=gs.array([0.5, 0.5]),
                point_b=gs.array([[0.5, -0.5], [0.4, 0.4]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_vectorization_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                point_a=gs.array([0.5, 0.5]),
                point_b=gs.array([[0.0, 0.0], [0.5, -0.5], [0.4, 0.4]]),
            )
        ]
        return self.generate_tests(smoke_data)
