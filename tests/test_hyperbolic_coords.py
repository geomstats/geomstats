"""Unit tests for the Hyperbolic space coordinates change.

We verify poincare ball model, poincare half plane
and minkowisky extrinsic/intrisic.
We also verify that converting point will lead to get same
distance (implemented for ball model and extrinsic only)
"""
import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_ball import PoincareBall


class TestHyperbolicCoords(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)
        self.dimension = 2

        self.extrinsic_manifold = Hyperboloid(
            dim=self.dimension)
        self.extrinsic_metric = self.extrinsic_manifold.metric

        self.ball_manifold = PoincareBall(
            dim=self.dimension)
        self.ball_metric = self.ball_manifold.metric

        self.intrinsic_manifold = Hyperboloid(
            dim=self.dimension, coords_type='intrinsic')
        self.intrinsic_metric = self.intrinsic_manifold.metric

        self.n_samples = 10

    def test_extrinsic_ball_extrinsic(self):
        x_in = gs.array([0.5, 7])
        x = self.intrinsic_manifold.to_coordinates(
            x_in, to_coords_type='extrinsic')
        x_b = self.extrinsic_manifold.to_coordinates(x, to_coords_type='ball')
        x2 = self.ball_manifold.to_coordinates(x_b, to_coords_type='extrinsic')
        self.assertAllClose(x, x2, atol=1e-8)

    def test_belongs_intrinsic(self):
        x_in = gs.array([0.5, 7])
        is_in = self.intrinsic_manifold.belongs(x_in)
        self.assertTrue(is_in)

    def test_belongs_extrinsic(self):
        x_true = self.intrinsic_manifold.to_coordinates(
            gs.array([0.5, 7]), 'extrinsic')
        x_false = gs.array([0.5, 7, 3.])
        is_in = self.extrinsic_manifold.belongs(x_true)
        self.assertTrue(is_in)
        is_out = self.extrinsic_manifold.belongs(x_false)
        self.assertFalse(is_out)

    def test_belongs_ball(self):
        x_true = gs.array([0.5, 0.5])
        x_false = gs.array([0.8, 0.8])
        is_in = self.ball_manifold.belongs(x_true)
        self.assertTrue(is_in)
        is_out = self.ball_manifold.belongs(x_false)
        self.assertFalse(is_out)

    def test_extrinsic_half_plane_extrinsic(self):
        x_in = gs.array([0.5, 7])
        x = self.intrinsic_manifold.to_coordinates(
            x_in, to_coords_type='extrinsic')
        x_up = self.extrinsic_manifold.to_coordinates(
            x, to_coords_type='half-plane')

        x2 = Hyperbolic.change_coordinates_system(
            x_up, 'half-plane', 'extrinsic')
        self.assertAllClose(x, x2, atol=1e-8)

    def test_intrinsic_extrinsic_intrinsic(self):
        x_intr = gs.array([0.5, 7])
        x_extr = self.intrinsic_manifold.to_coordinates(
            x_intr, to_coords_type='extrinsic')
        x_intr2 = self.extrinsic_manifold.to_coordinates(
            x_extr, to_coords_type='intrinsic')
        self.assertAllClose(x_intr, x_intr2, atol=1e-8)

    def test_ball_extrinsic_ball(self):
        x = gs.array([0.5, 0.2])
        x_e = self.ball_manifold.to_coordinates(x, to_coords_type='extrinsic')
        x2 = self.extrinsic_manifold.to_coordinates(x_e, to_coords_type='ball')
        self.assertAllClose(x, x2, atol=1e-10)

    def test_distance_ball_extrinsic_from_ball(self):
        x_ball = gs.array([0.7, 0.2])
        y_ball = gs.array([0.2, 0.2])
        x_extr = self.ball_manifold.to_coordinates(
            x_ball, to_coords_type='extrinsic')
        y_extr = self.ball_manifold.to_coordinates(
            y_ball, to_coords_type='extrinsic')
        dst_ball = self.ball_metric.dist(x_ball, y_ball)
        dst_extr = self.extrinsic_metric.dist(x_extr, y_extr)

        self.assertAllClose(dst_ball, dst_extr)

    def test_distance_ball_extrinsic_from_extr(self):
        x_int = gs.array([10, 0.2])
        y_int = gs.array([1, 6.])
        x_extr = self.intrinsic_manifold.to_coordinates(
            x_int, to_coords_type='extrinsic')
        y_extr = self.intrinsic_manifold.to_coordinates(
            y_int, to_coords_type='extrinsic')
        x_ball = self.extrinsic_manifold.to_coordinates(
            x_extr, to_coords_type='ball')
        y_ball = self.extrinsic_manifold.to_coordinates(
            y_extr, to_coords_type='ball')
        dst_ball = self.ball_metric.dist(x_ball, y_ball)
        dst_extr = self.extrinsic_metric.dist(x_extr, y_extr)

        self.assertAllClose(dst_ball, dst_extr)

    def test_distance_ball_extrinsic_from_extr_4_dim(self):
        x_int = gs.array([10, 0.2, 3, 4])
        y_int = gs.array([1, 6, 2., 1])

        ball_manifold = PoincareBall(4)
        extrinsic_manifold = Hyperboloid(4)

        ball_metric = ball_manifold.metric
        extrinsic_metric = extrinsic_manifold.metric

        x_extr = extrinsic_manifold.from_coordinates(
            x_int, from_coords_type='intrinsic')
        y_extr = extrinsic_manifold.from_coordinates(
            y_int, from_coords_type='intrinsic')
        x_ball = extrinsic_manifold.to_coordinates(
            x_extr, to_coords_type='ball')
        y_ball = extrinsic_manifold.to_coordinates(
            y_extr, to_coords_type='ball')
        dst_ball = ball_metric.dist(x_ball, y_ball)
        dst_extr = extrinsic_metric.dist(x_extr, y_extr)

        self.assertAllClose(dst_ball, dst_extr)

    def test_log_exp_ball_extrinsic_from_extr(self):
        """Compare log exp in different parameterizations."""
        x_int = gs.array([4., 0.2])
        y_int = gs.array([3., 3])
        x_extr = self.intrinsic_manifold.to_coordinates(
            x_int, to_coords_type='extrinsic')
        y_extr = self.intrinsic_manifold.to_coordinates(
            y_int, to_coords_type='extrinsic')
        x_ball = self.extrinsic_manifold.to_coordinates(
            x_extr, to_coords_type='ball')
        y_ball = self.extrinsic_manifold.to_coordinates(
            y_extr, to_coords_type='ball')

        x_ball_log_exp = self.ball_metric.exp(
            self.ball_metric.log(y_ball, x_ball), x_ball)

        x_extr_a = self.extrinsic_metric.exp(
            self.extrinsic_metric.log(y_extr, x_extr), x_extr)
        x_extr_b = self.extrinsic_manifold.from_coordinates(
            x_ball_log_exp, from_coords_type='ball')
        self.assertAllClose(x_extr_a, x_extr_b, atol=1e-4)

    def test_log_exp_ball(self):
        x = gs.array([0.1, 0.2])
        y = gs.array([0.2, 0.5])

        log = self.ball_metric.log(point=y, base_point=x)
        exp = self.ball_metric.exp(tangent_vec=log, base_point=x)
        self.assertAllClose(exp, y, atol=1e-1)

    def test_log_exp_ball_vectorization(self):
        x = gs.array([0.1, 0.2])
        y = gs.array([[0.2, 0.5], [0.1, 0.7]])

        log = self.ball_metric.log(y, x)
        exp = self.ball_metric.exp(log, x)
        self.assertAllClose(exp, y, atol=1e-1)

    def test_log_exp_ball_null_tangent(self):
        x = gs.array([[0.1, 0.2], [0.1, 0.2]])
        tangent_vec = gs.array([[0.0, 0.0], [0.0, 0.0]])
        exp = self.ball_metric.exp(tangent_vec, x)
        self.assertAllClose(exp, x, atol=1e-10)
