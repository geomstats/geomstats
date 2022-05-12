"""Unit tests for the Hyperbolic space using Poincare half space model."""

import geomstats.backend as gs
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_half_space import PoincareHalfSpaceMetric
from tests.conftest import Parametrizer, np_and_autograd_only
from tests.data.poincare_half_space_data import (
    PoincareHalfSpaceMetricTestData,
    PoincareHalfSpaceTestData,
)
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestPoincareHalfSpace(OpenSetTestCase, metaclass=Parametrizer):

    testing_data = PoincareHalfSpaceTestData()
    space = testing_data.space

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
