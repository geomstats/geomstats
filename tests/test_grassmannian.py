"""Unit tests for the Grassmannian."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.grassmannian import Grassmannian
from geomstats.geometry.grassmannian import GrassmannianCanonicalMetric

p_xy = gs.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 0.]])
p_yz = gs.array([
    [0., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]])
p_xz = gs.array([
    [1., 0., 0.],
    [0., 0., 0.],
    [0., 0., 1.]])

r_y = gs.array([
    [0., 0., 1.],
    [0., 0., 0.],
    [-1., 0., 0.]])
r_z = gs.array([
    [0., -1., 0.],
    [1., 0., 0.],
    [0., 0., 0.]])
pi_2 = gs.pi / 2
pi_4 = gs.pi / 4


class TestGrassmannian(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.n = 3
        self.k = 2
        self.space = Grassmannian(self.n, self.k)
        self.metric = GrassmannianCanonicalMetric(self.n, self.k)

    def test_exp_np(self):
        result = self.metric.exp(
            pi_2 * r_y,
            gs.array([p_xy, p_yz]))
        expected = gs.array([p_yz, p_xy])
        self.assertAllClose(result, expected)

        result = self.metric.exp(
            pi_2 * gs.array([r_y, r_z]),
            gs.array([p_xy, p_yz]))
        expected = gs.array([p_yz, p_xz])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_log(self):
        result = self.metric.log(
            self.metric.exp(pi_4 * r_y, p_xy),
            p_xy)
        expected = pi_4 * r_y
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_log_vectorized(self):
        tangent_vecs = pi_4 * gs.array([r_y, r_z])
        base_points = gs.array([p_xy, p_xz])
        points = self.metric.exp(tangent_vecs, base_points)
        result = self.metric.log(points, base_points)
        expected = tangent_vecs
        self.assertAllClose(result, expected)
