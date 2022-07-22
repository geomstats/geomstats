"""Unit tests for the Hyperbolic space."""

import geomstats.backend as gs
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hyperboloid import Hyperboloid, HyperboloidMetric
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.poincare_ball import PoincareBall
from tests.conftest import Parametrizer
from tests.data.hyperbolic_data import HyperbolicTestData, HyperboloidMetricTestData
from tests.geometry_test_cases import LevelSetTestCase, RiemannianMetricTestCase


class TestHyperbolic(LevelSetTestCase, metaclass=Parametrizer):
    space = Hyperboloid
    skip_test_extrinsic_after_intrinsic = True
    skip_test_projection_belongs = True

    testing_data = HyperbolicTestData()

    def test_belongs(self, dim, coords_type, vec, expected):
        space = self.space(dim, coords_type=coords_type)
        self.assertAllClose(space.belongs(vec), gs.array(expected))

    def test_regularize_raises(self, dim, point, expected):
        space = self.space(dim)
        with expected:
            space.regularize(point)

    def test_extrinsic_to_intrinsic_coords_rasises(self, dim, point, expected):
        space = self.space(dim)
        with expected:
            space.extrinsic_to_intrinsic_coords(point)

    def test_ball_extrinsic_ball(self, dim, x_ball):
        x_extrinsic = PoincareBall(dim).to_coordinates(
            x_ball, to_coords_type="extrinsic"
        )
        result = self.space(dim).to_coordinates(x_extrinsic, to_coords_type="ball")
        self.assertAllClose(result, x_ball)

    def test_extrinsic_ball_extrinsic_composition(self, dim, point_intrinsic):
        x = Hyperboloid(dim, coords_type="intrinsic").to_coordinates(
            point_intrinsic, to_coords_type="extrinsic"
        )
        x_b = Hyperboloid(dim).to_coordinates(x, to_coords_type="ball")
        x2 = PoincareBall(dim).to_coordinates(x_b, to_coords_type="extrinsic")
        self.assertAllClose(x, x2)

    def test_extrinsic_half_plane_extrinsic_composition(self, dim, point_intrinsic):
        x = Hyperboloid(dim, coords_type="intrinsic").to_coordinates(
            point_intrinsic, to_coords_type="extrinsic"
        )
        x_up = Hyperboloid(dim).to_coordinates(x, to_coords_type="half-space")
        x2 = Hyperbolic.change_coordinates_system(x_up, "half-space", "extrinsic")
        self.assertAllClose(x, x2)


class TestHyperboloidMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    connection = metric = HyperboloidMetric

    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = HyperboloidMetricTestData()

    def test_inner_product_is_minkowski_inner_product(
        self, dim, tangent_vec_a, tangent_vec_b, base_point
    ):
        metric = self.metric(dim)
        minkowki_space = Minkowski(dim + 1)
        result = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        expected = minkowki_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(result, expected)

    def test_scaled_inner_product(
        self, dim, scale, tangent_vec_a, tangent_vec_b, base_point
    ):
        default_space = Hyperboloid(dim=dim)
        scaled_space = Hyperboloid(dim=dim, scale=scale)
        inner_product_default_metric = default_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        inner_product_scaled_metric = scaled_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        result = inner_product_scaled_metric
        expected = scale**2 * inner_product_default_metric
        self.assertAllClose(result, expected)

    def test_scaled_squared_norm(self, dim, scale, tangent_vec, base_point):
        default_space = Hyperboloid(dim=dim)
        scaled_space = Hyperboloid(dim=dim, scale=scale)
        squared_norm_default_metric = default_space.metric.squared_norm(
            tangent_vec, base_point
        )
        squared_norm_scaled_metric = scaled_space.metric.squared_norm(
            tangent_vec, base_point
        )
        result = squared_norm_scaled_metric
        expected = scale**2 * squared_norm_default_metric
        self.assertAllClose(result, expected)

    def test_scaled_dist(self, dim, scale, point_a, point_b):
        default_space = Hyperboloid(dim=dim)
        scaled_space = Hyperboloid(dim=dim, scale=scale)
        distance_default_metric = default_space.metric.dist(point_a, point_b)
        distance_scaled_metric = scaled_space.metric.dist(point_a, point_b)
        result = distance_scaled_metric
        expected = scale * distance_default_metric
        self.assertAllClose(result, expected)

    def test_exp_after_log_intrinsic_ball_extrinsic(
        self, dim, x_intrinsic, y_intrinsic
    ):
        intrinsic_manifold = Hyperboloid(dim=dim, coords_type="intrinsic")
        extrinsic_manifold = Hyperbolic(dim=dim, coords_type="extrinsic")
        ball_manifold = PoincareBall(dim)
        x_extr = intrinsic_manifold.to_coordinates(
            x_intrinsic, to_coords_type="extrinsic"
        )
        y_extr = intrinsic_manifold.to_coordinates(
            y_intrinsic, to_coords_type="extrinsic"
        )
        x_ball = extrinsic_manifold.to_coordinates(x_extr, to_coords_type="ball")
        y_ball = extrinsic_manifold.to_coordinates(y_extr, to_coords_type="ball")

        x_ball_exp_after_log = ball_manifold.metric.exp(
            ball_manifold.metric.log(y_ball, x_ball), x_ball
        )

        x_extr_a = extrinsic_manifold.metric.exp(
            extrinsic_manifold.metric.log(y_extr, x_extr), x_extr
        )
        x_extr_b = extrinsic_manifold.from_coordinates(
            x_ball_exp_after_log, from_coords_type="ball"
        )
        self.assertAllClose(x_extr_a, x_extr_b, atol=3e-4)

    def test_distance_ball_extrinsic_from_ball(self, dim, x_ball, y_ball):

        ball_manifold = PoincareBall(dim)
        space = Hyperboloid(dim)
        x_extr = ball_manifold.to_coordinates(x_ball, to_coords_type="extrinsic")
        y_extr = ball_manifold.to_coordinates(y_ball, to_coords_type="extrinsic")
        dst_ball = ball_manifold.metric.dist(x_ball, y_ball)
        dst_extr = space.metric.dist(x_extr, y_extr)
        self.assertAllClose(dst_ball, dst_extr)

    def test_distance_ball_extrinsic_intrinsic(self, dim, x_intrinsic, y_intrinsic):

        intrinsic_manifold = Hyperboloid(dim, coords_type="intrinsic")
        extrinsic_manifold = Hyperboloid(dim, coords_type="extrinsic")
        x_extr = intrinsic_manifold.to_coordinates(
            x_intrinsic, to_coords_type="extrinsic"
        )
        y_extr = intrinsic_manifold.to_coordinates(
            y_intrinsic, to_coords_type="extrinsic"
        )
        x_ball = extrinsic_manifold.to_coordinates(x_extr, to_coords_type="ball")
        y_ball = extrinsic_manifold.to_coordinates(y_extr, to_coords_type="ball")
        dst_ball = PoincareBall(dim).metric.dist(x_ball, y_ball)
        dst_extr = extrinsic_manifold.metric.dist(x_extr, y_extr)

        self.assertAllClose(dst_ball, dst_extr)
