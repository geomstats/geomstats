"""
Unit tests for the Grassmannian.
"""

import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry.grassmannian import Grassmannian
from geomstats.geometry.grassmannian import GrassmannianCanonicalMetric

Pxy = gs.diag([1., 1., 0.])[0]
Pyz = gs.diag([0., 1., 1.])[0]
Pxz = gs.diag([1., 0., 1.])[0]
Ry = gs.array([
    [0., 0., 1.],
    [0., 0., 0.],
    [-1., 0., 0.]])
Rz = gs.array([
    [0., -1., 0.],
    [1., 0., 0.],
    [0., 0., 0.]])
pi_2 = gs.pi/2


class TestGrassmannianMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)

        self.n = 3
        self.k = 2
        self.space = Grassmannian(self.n, self.k)
        self.metric = GrassmannianCanonicalMetric(self.n, self.k)


    def test_exp(self): 
        result = self.metric.exp(pi_2 * Ry, [Pxy, Pyz])
        expected = gs.array([Pyz, Pxy])
        self.assertAllClose(result, expected)

        result = self.metric.exp(
            pi_2 * gs.array([Ry, Rz]), 
            gs.array([Pxy, Pyz]))
        expected = gs.array([Pyz, Pxz])
        self.assertAllClose(result, expected)

if __name__ == '__main__':
        geomstats.test.main()
