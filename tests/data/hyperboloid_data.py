import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.hyperboloid import Hyperboloid, HyperboloidMetric
from tests.data_generation import _LevelSetTestData, _RiemannianMetricTestData

# Tolerance for errors on predicted vectors, relative to the *norm*
# of the vector, as opposed to the standard behavior of gs.allclose
# where it is relative to each element of the array

RTOL = 1e-6


class HyperboloidTestData(_LevelSetTestData):

    dim_list = random.sample(range(2, 4), 2)
    space_args_list = [(dim,) for dim in dim_list]
    shape_list = [(dim + 1,) for dim in dim_list]
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = Hyperboloid

    def belongs_test_data(self):
        smoke_data = [
            dict(
                dim=3,
                default_coords_type="extrinsic",
                vec=gs.array([1.0, 0.0, 0.0, 0.0]),
                expected=True,
            ),
            dict(
                dim=2,
                default_coords_type="extrinsic",
                vec=gs.array([0.5, 7, 3.0]),
                expected=False,
            ),
            dict(
                dim=2,
                default_coords_type="intrinsic",
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

    def extrinsic_ball_extrinsic_composition_test_data(self):
        smoke_data = [dict(dim=2, point_intrinsic=gs.array([0.5, 7]))]
        return self.generate_tests(smoke_data)

    def extrinsic_half_plane_extrinsic_composition_test_data(self):
        smoke_data = [dict(dim=2, point_intrinsic=gs.array([0.5, 7], dtype=gs.float64))]
        return self.generate_tests(smoke_data)

    def ball_extrinsic_ball_test_data(self):
        smoke_data = [dict(dim=2, x_ball=gs.array([0.5, 0.2]))]
        return self.generate_tests(smoke_data)


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

    Metric = HyperboloidMetric

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

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=10.0)

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
