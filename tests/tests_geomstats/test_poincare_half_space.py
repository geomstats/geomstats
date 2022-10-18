"""Unit tests for the Hyperbolic space using Poincare half space model."""

import geomstats.backend as gs
from geomstats.geometry.hyperboloid import Hyperboloid
from tests.conftest import Parametrizer, np_and_autograd_only
from tests.data.poincare_half_space_data import (
    PoincareHalfSpaceMetricTestData,
    PoincareHalfSpaceTestData,
)
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestPoincareHalfSpace(OpenSetTestCase, metaclass=Parametrizer):

    testing_data = PoincareHalfSpaceTestData()

    def test_belongs(self, dim, vec, expected):
        space = self.Space(dim)
        self.assertAllClose(space.belongs(gs.array(vec)), gs.array(expected))

    def test_half_space_to_ball_coordinates(self, dim, point, expected):
        space = self.Space(dim)
        result = space.half_space_to_ball_coordinates(gs.array(point))
        self.assertAllClose(result, gs.array(expected))

    def test_half_space_coordinates_ball_coordinates_composition(
        self, dim, point_half_space
    ):
        space = self.Space(dim)
        point_ball = space.half_space_to_ball_coordinates(point_half_space)
        result = space.ball_to_half_space_coordinates(point_ball)
        self.assertAllClose(result, point_half_space)

    def test_ball_half_plane_tangent_are_inverse(self, dim, tangent_vec, base_point):
        space = self.Space(dim)
        tangent_vec_ball = space.half_space_to_ball_tangent(tangent_vec, base_point)
        base_point_ball = space.half_space_to_ball_coordinates(base_point)
        result = space.ball_to_half_space_tangent(tangent_vec_ball, base_point_ball)
        self.assertAllClose(result, tangent_vec)

    def test_ball_to_half_space_coordinates(self, dim, point_ball):
        space = self.Space(dim)
        point_half_space = space.ball_to_half_space_coordinates(point_ball)
        point_ext = Hyperboloid(dim).from_coordinates(point_ball, "ball")
        point_half_space_expected = Hyperboloid(dim).to_coordinates(
            point_ext, "half-space"
        )
        self.assertAllClose(point_half_space, point_half_space_expected)


class TestPoincareHalfSpaceMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_belongs = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = PoincareHalfSpaceMetricTestData()

    def test_inner_product(
        self, dim, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        metric = self.Metric(dim)
        result = metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_exp_and_coordinates_tangent(self, dim, tangent_vec, base_point):
        metric = self.Metric(dim)
        end_point = metric.exp(tangent_vec, base_point)
        self.assertAllClose(base_point[0], end_point[0])

    @np_and_autograd_only
    def test_exp(self, dim, tangent_vec, base_point, expected):
        metric = self.Metric(dim)
        result = metric.exp(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))
