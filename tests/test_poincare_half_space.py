"""Unit tests for the Hyperbolic space using Poincare half space model."""
import numpy as np

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_half_space import PoincareHalfSpace


class TestPoincareHalfSpace(geomstats.tests.TestCase):
    def setUp(self):
        self.manifold = PoincareHalfSpace(2)
        self.metric = self.manifold.metric

        self.hyperboloid_manifold = Hyperboloid(2)
        self.hyperboloid_metric = self.hyperboloid_manifold.metric

    def test_half_space_to_ball_coordinates(self):
        point_half_space = gs.array([1.5, 2.3])
        point_ball = self.metric.half_space_to_ball_coordinates(
            point_half_space)
        point_ext = self.hyperboloid_manifold.from_coordinates(
            point_half_space, 'half-plane')
        point_ball_expected = self.hyperboloid_manifold.to_coordinates(
            point_ext, 'ball')
        self.assertAllClose(point_ball, point_ball_expected)

    def test_ball_to_half_space_coordinates(self):
        point_ball = gs.array([-0.3, 0.7])
        point_half_space = self.metric.ball_to_half_space_coordinates(
            point_ball)
        point_ext = self.hyperboloid_manifold.from_coordinates(
            point_ball, 'ball')
        point_half_space_expected = self.hyperboloid_manifold.to_coordinates(
            point_ext, 'half-plane')
        self.assertAllClose(point_half_space, point_half_space_expected)

    def test_coordinates(self):
        point_half_space = gs.array([1.5, 2.3])
        point_ball = self.metric.half_space_to_ball_coordinates(
            point_half_space)
        result = self.metric.ball_to_half_space_coordinates(
            point_ball)
        self.assertAllClose(result, point_half_space)

    def test_exp_and_coordinates_tangent(self):
        base_point = gs.array([1.5, 2.3])
        tangent_vec = gs.array([0., 1.])
        end_point = self.metric.exp(tangent_vec, base_point)
        self.assertAllClose(base_point[0], end_point[0])

    @geomstats.tests.np_only
    def test_exp(self):
        point = gs.array([1., 1.])
        tangent_vec = gs.array([2., 1.])
        end_point = self.metric.exp(tangent_vec, point)

        circle_center = point[0] + point[1] * tangent_vec[1] / tangent_vec[0]
        circle_radius = gs.sqrt((circle_center - point[0])**2 + point[1]**2)

        moebius_d = 1
        moebius_c = 1 / (2 * circle_radius)
        moebius_b = circle_center - circle_radius
        moebius_a = (circle_center + circle_radius) * moebius_c

        point_complex = point[0] + 1j * point[1]
        tangent_vec_complex = tangent_vec[0] + 1j * tangent_vec[1]

        point_moebius = 1j * (moebius_d * point_complex - moebius_b)\
            / (moebius_c * point_complex - moebius_a)
        tangent_vec_moebius = -1j * tangent_vec_complex * (
            1j * moebius_c * point_moebius + moebius_d)**2

        end_point_moebius = point_moebius * gs.exp(
            tangent_vec_moebius / point_moebius)
        end_point_complex = (moebius_a * 1j * end_point_moebius + moebius_b)\
            / (moebius_c * 1j * end_point_moebius + moebius_d)
        end_point_expected = gs.hstack(
            [np.real(end_point_complex), np.imag(end_point_complex)])

        self.assertAllClose(end_point, end_point_expected)
