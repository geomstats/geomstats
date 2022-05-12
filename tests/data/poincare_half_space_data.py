import random

import numpy as np

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData


class PoincareHalfSpaceTestData(_OpenSetTestData):
    dim_list = random.sample(range(2, 5), 2)
    space_args_list = [(dim,) for dim in dim_list]
    shape_list = [(dim,) for dim in dim_list]
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    space = PoincareHalfSpace

    def belongs_test_data(self):
        smoke_data = [
            dict(dim=2, vec=[1.5, 2.3], expected=True),
            dict(dim=2, vec=[[1.5, 2.0], [2.5, -0.3]], expected=[True, False]),
        ]
        return self.generate_tests(smoke_data)

    def half_space_to_ball_coordinates_test_data(self):
        smoke_data = [
            dict(dim=2, point=[0.0, 1.0], expected=gs.zeros(2)),
            dict(
                dim=2,
                point=[[0.0, 1.0], [0.0, 2.0]],
                expected=[[0.0, 0.0], [0.0, 1.0 / 3.0]],
            ),
        ]
        return self.generate_tests(smoke_data)

    def ball_half_plane_tangent_are_inverse_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                tangent_vec=gs.array([0.5, 1.0]),
                base_point=gs.array([1.5, 2.3]),
            )
        ]
        return self.generate_tests(smoke_data)

    def ball_to_half_space_coordinates_test_data(self):
        smoke_data = [dict(dim=2, point_ball=gs.array([-0.3, 0.7]))]
        return self.generate_tests(smoke_data)

    def half_space_coordinates_ball_coordinates_composition_test_data(self):
        smoke_data = [dict(dim=2, point_half_space=gs.array([1.5, 2.3]))]
        return self.generate_tests(smoke_data)


class PoincareHalfSpaceMetricTestData(_RiemannianMetricTestData):
    dim_list = random.sample(range(2, 5), 2)
    metric_args_list = [(dim,) for dim in dim_list]
    shape_list = [(dim,) for dim in dim_list]
    space_list = [PoincareHalfSpace(dim) for dim in dim_list]
    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                tangent_vec_a=[[1.0, 2.0], [3.0, 4.0]],
                tangent_vec_b=[[1.0, 2.0], [3.0, 4.0]],
                base_point=[[0.0, 1.0], [0.0, 5.0]],
                expected=[5.0, 1.0],
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_and_coordinates_tangent_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                tangent_vec=gs.array([0.0, 1.0]),
                base_point=gs.array([1.5, 2.3]),
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        def _exp(tangent_vec, base_point):
            circle_center = (
                base_point[0] + base_point[1] * tangent_vec[1] / tangent_vec[0]
            )
            circle_radius = gs.sqrt(
                (circle_center - base_point[0]) ** 2 + base_point[1] ** 2
            )

            moebius_d = 1
            moebius_c = 1 / (2 * circle_radius)
            moebius_b = circle_center - circle_radius
            moebius_a = (circle_center + circle_radius) * moebius_c

            point_complex = base_point[0] + 1j * base_point[1]
            tangent_vec_complex = tangent_vec[0] + 1j * tangent_vec[1]

            point_moebius = (
                1j
                * (moebius_d * point_complex - moebius_b)
                / (moebius_c * point_complex - moebius_a)
            )
            tangent_vec_moebius = (
                -1j
                * tangent_vec_complex
                * (1j * moebius_c * point_moebius + moebius_d) ** 2
            )

            end_point_moebius = point_moebius * gs.exp(
                tangent_vec_moebius / point_moebius
            )
            end_point_complex = (moebius_a * 1j * end_point_moebius + moebius_b) / (
                moebius_c * 1j * end_point_moebius + moebius_d
            )
            end_point_expected = gs.hstack(
                [np.real(end_point_complex), np.imag(end_point_complex)]
            )
            return end_point_expected

        inputs_to_exp = [(gs.array([2.0, 1.0]), gs.array([1.0, 1.0]))]
        smoke_data = []
        if not geomstats.tests.tf_backend():
            for tangent_vec, base_point in inputs_to_exp:
                smoke_data.append(
                    dict(
                        dim=2,
                        tangent_vec=tangent_vec,
                        base_point=base_point,
                        expected=_exp(tangent_vec, base_point),
                    )
                )
        return self.generate_tests(smoke_data)

    def exp_shape_test_data(self):
        return self._exp_shape_test_data(
            self.metric_args_list, self.space_list, self.shape_list
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
            is_tangent_atol=gs.atol * 10900,
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
        )

    def exp_geodesic_ivp_test_data(self):
        return self._exp_geodesic_ivp_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            self.n_points_list,
            rtol=gs.rtol * 100000,
            atol=gs.atol * 100000,
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
            self.n_tangent_list,
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

    def triangle_inequality_of_dist_test_data(self):
        return self._triangle_inequality_of_dist_test_data(
            self.metric_args_list, self.space_list, self.n_points_list
        )

    def inner_product_is_symmetric_test_data(self):
        return self._inner_product_is_symmetric_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
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
