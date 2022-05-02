"""Unit tests for the functions manifolds."""

import numpy as np

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.functions import HilbertSphere


def gaussian(x, mu, sig):
    a = (x - mu) ** 2 / (2 * (sig**2))
    b = 1 / (sig * (gs.sqrt(2 * gs.pi)))
    f = (b * gs.exp(-a)).reshape(1, len(x))
    l2_norm = gs.sqrt(np.trapz(f**2, x, axis=1))
    f_sinf = f / l2_norm

    return f_sinf


class TestHilbertSphereMetric(geomstats.tests.TestCase):
    def setup_method(self):
        self.domain = gs.linspace(0, 1, num=50)
        self.f = gaussian(self.domain, 0.5, 0.1)
        self.manifold = HilbertSphere(self.domain)
        self.point_a = gaussian(self.domain, 0.2, 0.1)
        self.point_b = gaussian(self.domain, 0.5, 0.1)

    @geomstats.tests.np_and_autograd_only
    def test_inner_product(self):
        result = self.manifold.metric.inner_product(self.point_a, self.point_b)
        self.assertAllClose(gs.shape(result), (1,))

    @geomstats.tests.np_and_autograd_only
    def test_exp(self):
        exp = self.manifold.metric.exp(self.point_a, self.point_b)
        self.assertAllClose(gs.shape(exp), gs.shape(self.point_b))
        result = self.manifold.belongs(exp, atol=0.1)[0]
        self.assertTrue(result, "Expected True but got %s" % result)

    @geomstats.tests.np_and_autograd_only
    def test_log(self):
        log = self.manifold.metric.log(self.point_a, self.point_b)
        self.assertAllClose(gs.shape(log), gs.shape(self.point_b))


class TestHilbertSphere(geomstats.tests.TestCase):
    def setup_method(self):
        self.domain = gs.linspace(0, 1, num=50)
        self.f = gaussian(self.domain, 0.5, 0.1)
        self.manifold = HilbertSphere(self.domain)
        self.point_a = gaussian(self.domain, 0.2, 0.1)
        self.point_b = gaussian(self.domain, 0.5, 0.2)

    @geomstats.tests.np_and_autograd_only
    def test_projection(self):
        result = self.manifold.belongs(self.f)
        self.assertTrue(result, "Expected True but got %s" % result)

    @geomstats.tests.np_and_autograd_only
    def test_belongs(self):
        result = self.manifold.belongs(self.point_a)
        self.assertTrue(result[0], "Expected True but got %s" % result[0])

        f = gs.sin(gs.linspace(-gs.pi, gs.pi, 50)).reshape(1, 50)
        result = self.manifold.belongs(f)
        self.assertFalse(result[0], "Expected False but got %s" % result[0])

    @geomstats.tests.np_and_autograd_only
    def test_to_tangent(self):
        tangent_vec = self.manifold.to_tangent(self.point_b, self.point_a)
        self.assertAllClose(gs.shape(tangent_vec), gs.shape(self.point_b))

    @geomstats.tests.np_and_autograd_only
    def test_is_tangent(self):
        tangent_vec = self.manifold.to_tangent(self.point_a, self.point_b)
        result = self.manifold.is_tangent(tangent_vec, self.point_b)
        self.assertTrue(result)

    @geomstats.tests.np_and_autograd_only
    def test_random_point(self):
        rand_point = self.manifold.random_point()
        result = self.manifold.belongs(rand_point)
        self.assertTrue(result)
