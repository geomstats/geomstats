"""Unit tests for the Hyperbolic space using Poincare half space model."""

import random

import numpy as np

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_half_space import (
    PoincareHalfSpace,
    PoincareHalfSpaceMetric,
)
from tests.conftest import Parametrizer, np_and_autograd_only
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestPoincareHalfSpace(OpenSetTestCase, metaclass=Parametrizer):
    space = PoincareHalfSpace

    class PoincareHalfSpaceTestData(_OpenSetTestData):
        dim_list = random.sample(range(2, 5), 2)
        space_args_list = [(dim,) for dim in dim_list]
        shape_list = [(dim,) for dim in dim_list]
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

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

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(2,), (3,)]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list, self.shape_list, self.n_points_list
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                PoincareHalfSpace,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def to_tangent_is_tangent_in_ambient_space_test_data(self):
            return self._to_tangent_is_tangent_in_ambient_space_test_data(
                PoincareHalfSpace, self.space_args_list, self.shape_list
            )

    testing_data = PoincareHalfSpaceTestData()

    def test_belongs(self, dim, vec, expected):
        space = self.space(dim)
        self.assertAllClose(space.belongs(gs.array(vec)), gs.array(expected))

    def test_half_space_to_ball_coordinates(self, dim, point, expected):
        space = self.space(dim)
        result = space.half_space_to_ball_coordinates(gs.array(point))
        self.assertAllClose(result, gs.array(expected))

    def test_half_space_coordinates_ball_coordinates_composition(
        self, dim, point_half_space
    ):
        space = self.space(dim)
        point_ball = space.half_space_to_ball_coordinates(point_half_space)
        result = space.ball_to_half_space_coordinates(point_ball)
        self.assertAllClose(result, point_half_space)

    def test_ball_half_plane_tangent_are_inverse(self, dim, tangent_vec, base_point):
        space = self.space(dim)
        tangent_vec_ball = space.half_space_to_ball_tangent(tangent_vec, base_point)
        base_point_ball = space.half_space_to_ball_coordinates(base_point)
        result = space.ball_to_half_space_tangent(tangent_vec_ball, base_point_ball)
        self.assertAllClose(result, tangent_vec)

    def test_ball_to_half_space_coordinates(self, dim, point_ball):
        space = self.space(dim)
        point_half_space = space.ball_to_half_space_coordinates(point_ball)
        point_ext = Hyperboloid(dim).from_coordinates(point_ball, "ball")
        point_half_space_expected = Hyperboloid(dim).to_coordinates(
            point_ext, "half-space"
        )
        self.assertAllClose(point_half_space, point_half_space_expected)


class TestPoincareHalfSpaceMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = PoincareHalfSpaceMetric
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_belongs = True

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

        def log_then_exp_test_data(self):
            return self._log_then_exp_test_data(
                self.metric_args_list,
                self.space_list,
                self.n_points_list,
                rtol=gs.rtol * 100,
                atol=gs.atol * 10000,
            )

        def exp_then_log_test_data(self):
            return self._exp_then_log_test_data(
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

    testing_data = PoincareHalfSpaceMetricTestData()

    def test_inner_product(
        self, dim, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        metric = self.metric(dim)
        result = metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_exp_and_coordinates_tangent(self, dim, tangent_vec, base_point):
        metric = self.metric(dim)
        end_point = metric.exp(tangent_vec, base_point)
        self.assertAllClose(base_point[0], end_point[0])

    @np_and_autograd_only
    def test_exp(self, dim, tangent_vec, base_point, expected):
        metric = self.metric(dim)
        result = metric.exp(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))
