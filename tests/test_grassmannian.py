"""
Unit tests for the Grassmannian.
"""

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


class TestGrassmannianMethods(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.n = 3
        self.k = 2
        self.space = Grassmannian(self.n, self.k)
        self.metric = GrassmannianCanonicalMetric(self.n, self.k)

    @geomstats.tests.np_only
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
