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

    def test_belongs(self):
        point = gs.array([1.5, 2.3])
        result = self.manifold.belongs(point)
        self.assertTrue(result)

        points = gs.array([[1.5, 2.], [2.5, -0.3]])
        result = self.manifold.belongs(points)
        expected = gs.array([True, False])
        self.assertAllClose(result, expected)

    def test_inner_product_vectorization(self):
        tangent_vec = gs.array([[1., 2.], [3., 4.]])
        base_point = gs.array([[0., 1.], [0., 5.]])
        result = self.metric.inner_product(
            tangent_vec, tangent_vec, base_point)
        expected = gs.array([5., 1.])
        self.assertAllClose(result, expected)

    def test_half_space_to_ball_coordinates(self):
        point_half_space = gs.array([0., 1.])
        result = self.manifold.half_space_to_ball_coordinates(
            point_half_space)
        expected = gs.zeros(2)
        self.assertAllClose(result, expected)

    def test_half_space_to_ball_coordinates_vectorization(self):
        point_half_space = gs.array([[0., 1.], [0., 2.]])
        point_ball = self.manifold.half_space_to_ball_coordinates(
            point_half_space)
        expected = gs.array([[0., 0.], [0., 1. / 3.]])
        self.assertAllClose(point_ball, expected)

    def test_ball_to_half_space_coordinates(self):
        point_ball = gs.array([-0.3, 0.7])
        point_half_space = self.manifold.ball_to_half_space_coordinates(
            point_ball)
        point_ext = self.hyperboloid_manifold.from_coordinates(
            point_ball, 'ball')
        point_half_space_expected = self.hyperboloid_manifold.to_coordinates(
            point_ext, 'half-space')
        self.assertAllClose(point_half_space, point_half_space_expected)

    def test_coordinates(self):
        point_half_space = gs.array([1.5, 2.3])
        point_ball = self.manifold.half_space_to_ball_coordinates(
            point_half_space)
        result = self.manifold.ball_to_half_space_coordinates(
            point_ball)
        self.assertAllClose(result, point_half_space)

    def test_exp_and_coordinates_tangent(self):
        base_point = gs.array([1.5, 2.3])
        tangent_vec = gs.array([0., 1.])
        end_point = self.metric.exp(tangent_vec, base_point)
        self.assertAllClose(base_point[0], end_point[0])

    def test_ball_half_plane_are_inverse(self):
        base_point = gs.array([1.5, 2.3])
        base_point_ball = self.manifold.half_space_to_ball_coordinates(
            base_point)
        result = self.manifold.ball_to_half_space_coordinates(
            base_point_ball)
        self.assertAllClose(result, base_point)

    def test_ball_half_plane_tangent_are_inverse(self):
        base_point = gs.array([1.5, 2.3])
        tangent_vec = gs.array([0.5, 1.])
        tangent_vec_ball = self.manifold.half_space_to_ball_tangent(
            tangent_vec, base_point)
        base_point_ball = self.manifold.half_space_to_ball_coordinates(
            base_point)
        result = self.manifold.ball_to_half_space_tangent(
            tangent_vec_ball, base_point_ball)
        self.assertAllClose(result, tangent_vec)

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

    @geomstats.tests.np_only
    def test_exp_vectorization(self):
        point = gs.array([[1., 1.], [1., 1.]])
        tangent_vec = gs.array([[2., 1.], [2., 1.]])
        result = self.metric.exp(tangent_vec, point)

        point = point[0]
        tangent_vec = tangent_vec[0]
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
        expected = gs.stack([end_point_expected, end_point_expected])
        self.assertAllClose(result, expected)

    def test_exp_and_log_are_inverse(self):
        points = gs.array([[1., 1.], [1., 1.]])
        tangent_vecs = gs.array([[2., 1.], [2., 1.]])
        end_points = self.metric.exp(tangent_vecs, points)
        result = self.metric.log(end_points, points)
        expected = tangent_vecs
        self.assertAllClose(result, expected)

    def test_projection(self):
        point = gs.array([[1., -1.], [0., 1.]])
        projected = self.manifold.projection(point)
        result = self.manifold.belongs(projected)
        self.assertTrue(gs.all(result))

        projected = self.manifold.projection(point[0])
        result = self.manifold.belongs(projected)
        self.assertTrue(result)
