import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hyperboloid import Hyperboloid
from tests.data_generation import _LevelSetTestData, _RiemannianMetricTestData

# Tolerance for errors on predicted vectors, relative to the *norm*
# of the vector, as opposed to the standard behavior of gs.allclose
# where it is relative to each element of the array

RTOL = 1e-6


class HyperbolicTestData(_LevelSetTestData):

    dim_list = random.sample(range(2, 4), 2)
    space_args_list = [(dim,) for dim in dim_list]
    shape_list = [(dim + 1,) for dim in dim_list]
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    def belongs_test_data(self):
        smoke_data = [
            dict(
                dim=3,
                coords_type="extrinsic",
                vec=gs.array([1.0, 0.0, 0.0, 0.0]),
                expected=True,
            ),
            dict(
                dim=2,
                coords_type="extrinsic",
                vec=gs.array([0.5, 7, 3.0]),
                expected=False,
            ),
            dict(
                dim=2,
                coords_type="intrinsic",
                vec=gs.array([0.5, 7]),
                expected=True,
            ),
        ]
        return self.generate_tests(smoke_data)

    def regularize_raises_test_data(self):
        smoke_data = [
            dict(
                dim=3,
                point=gs.array([-1.0, 1.0, 0.0, 0.0]),
                expected=pytest.raises(ValueError),
            )
        ]
        return self.generate_tests(smoke_data)

    def extrinsic_to_intrinsic_coords_rasises_test_data(self):
        smoke_data = [
            dict(
                dim=3,
                point=gs.array([-1.0, 1.0, 0.0, 0.0]),
                expected=pytest.raises(ValueError),
            )
        ]
        return self.generate_tests(smoke_data)

    def random_point_belongs_test_data(self):
        smoke_space_args_list = [(2,), (3,)]
        smoke_n_points_list = [1, 2]
        belongs_atol = gs.atol * 100000
        return self._random_point_belongs_test_data(
            smoke_space_args_list,
            smoke_n_points_list,
            self.space_args_list,
            self.n_points_list,
            belongs_atol,
        )

    def to_tangent_is_tangent_test_data(self):

        is_tangent_atol = gs.atol * 100000

        return self._to_tangent_is_tangent_test_data(
            Hyperboloid,
            self.space_args_list,
            self.shape_list,
            self.n_vecs_list,
            is_tangent_atol,
        )

    def projection_belongs_test_data(self):
        return self._projection_belongs_test_data(
            self.space_args_list,
            self.shape_list,
            self.n_points_list,
            belongs_atol=gs.atol * 100000,
        )

    def intrinsic_after_extrinsic_test_data(self):
        return self._intrinsic_after_extrinsic_test_data(
            Hyperbolic, self.space_args_list, self.n_points_list
        )

    def extrinsic_after_intrinsic_test_data(self):
        return self._extrinsic_after_intrinsic_test_data(
            Hyperbolic, self.space_args_list, self.n_points_list
        )

    def extrinsic_ball_extrinsic_composition_test_data(self):
        smoke_data = [dict(dim=2, point_intrinsic=gs.array([0.5, 7]))]
        return self.generate_tests(smoke_data)

    def extrinsic_half_plane_extrinsic_composition_test_data(self):
        smoke_data = [dict(dim=2, point_intrinsic=gs.array([0.5, 7], dtype=gs.float64))]
        return self.generate_tests(smoke_data)

    def ball_extrinsic_ball_test_data(self):
        smoke_data = [dict(dim=2, x_ball=gs.array([0.5, 0.2]))]
        return self.generate_tests(smoke_data)

    def random_tangent_vec_is_tangent_test_data(self):
        return self._random_tangent_vec_is_tangent_test_data(
            Hyperbolic, self.space_args_list, self.n_vecs_list
        )


class HyperboloidMetricTestData(_RiemannianMetricTestData):

    dim_list = random.sample(range(2, 4), 2)
    metric_args_list = [(dim,) for dim in dim_list]
    shape_list = [(dim + 1,) for dim in dim_list]
    space_list = [Hyperboloid(dim) for dim in dim_list]
    n_points_list = random.sample(range(1, 5), 2)
    n_tangent_vecs_list = random.sample(range(1, 5), 2)
    n_points_a_list = random.sample(range(1, 5), 2)
    n_points_b_list = [1]
    alpha_list = [1] * 2
    n_rungs_list = [1] * 2
    scheme_list = ["pole"] * 2

    def inner_product_is_minkowski_inner_product_test_data(self):
        space = Hyperboloid(dim=3)
        base_point = gs.array([1.16563816, 0.36381045, -0.47000603, 0.07381469])
        tangent_vec_a = space.to_tangent(
            vector=gs.array([10.0, 200.0, 1.0, 1.0]), base_point=base_point
        )
        tangent_vec_b = space.to_tangent(
            vector=gs.array([11.0, 20.0, -21.0, 0.0]), base_point=base_point
        )
        smoke_data = [
            dict(
                dim=3,
                tangent_vec_a=tangent_vec_a,
                tangent_vec_b=tangent_vec_b,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def scaled_inner_product_test_data(self):
        space = Hyperboloid(3)
        base_point = space.from_coordinates(gs.array([1.0, 1.0, 1.0]), "intrinsic")
        tangent_vec_a = space.to_tangent(gs.array([1.0, 2.0, 3.0, 4.0]), base_point)
        tangent_vec_b = space.to_tangent(gs.array([5.0, 6.0, 7.0, 8.0]), base_point)
        smoke_data = [
            dict(
                dim=3,
                scale=2,
                tangent_vec_a=tangent_vec_a,
                tangent_vec_b=tangent_vec_b,
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)

    def scaled_squared_norm_test_data(self):
        space = Hyperboloid(3)
        base_point = space.from_coordinates(gs.array([1.0, 1.0, 1.0]), "intrinsic")
        tangent_vec = space.to_tangent(gs.array([1.0, 2.0, 3.0, 4.0]), base_point)
        smoke_data = [
            dict(dim=3, scale=2, tangent_vec=tangent_vec, base_point=base_point)
        ]
        return self.generate_tests(smoke_data)

    def scaled_dist_test_data(self):
        space = Hyperboloid(3)
        point_a = space.from_coordinates(gs.array([1.0, 2.0, 3.0]), "intrinsic")
        point_b = space.from_coordinates(gs.array([4.0, 5.0, 6.0]), "intrinsic")
        smoke_data = [dict(dim=3, scale=2, point_a=point_a, point_b=point_b)]
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
            belongs_atol=1e-2,
        )

    def log_is_tangent_test_data(self):
        return self._log_is_tangent_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            is_tangent_atol=1e-2,
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
            atol=gs.atol * 100000,
        )

    def log_after_exp_test_data(self):
        return self._log_after_exp_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            amplitude=10.0,
            rtol=gs.rtol * 100,
            atol=gs.atol * 100000,
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
            is_tangent_atol=gs.atol * 10000,
            rtol=gs.rtol * 100,
            atol=gs.atol * 10000,
        )

    def parallel_transport_bvp_is_isometry_test_data(self):
        return self._parallel_transport_bvp_is_isometry_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            is_tangent_atol=gs.atol * 10000,
            rtol=gs.rtol * 100,
            atol=gs.atol * 10000,
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

    def exp_after_log_intrinsic_ball_extrinsic_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                x_intrinsic=gs.array([4.0, 0.2]),
                y_intrinsic=gs.array([3.0, 3]),
            )
        ]
        return self.generate_tests(smoke_data)

    def distance_ball_extrinsic_from_ball_test_data(self):

        smoke_data = [
            dict(dim=2, x_ball=gs.array([0.7, 0.2]), y_ball=gs.array([0.2, 0.2]))
        ]
        return self.generate_tests(smoke_data)

    def distance_ball_extrinsic_intrinsic_test_data(self):
        smoke_data = [
            dict(
                dim=2,
                x_intrinsic=gs.array([10, 0.2]),
                y_intrinsic=gs.array([1, 6.0]),
            ),
            dict(
                dim=4,
                x_intrinsic=gs.array([10, 0.2, 3, 4]),
                y_intrinsic=gs.array([1, 6, 2.0, 1]),
            ),
        ]
        return self.generate_tests(smoke_data)
