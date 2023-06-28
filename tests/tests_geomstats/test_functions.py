"""Unit tests for the functions manifolds."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.functions import HilbertSphere


def gaussian(x, mu, sig):
    a = (x - mu) ** 2 / (2 * (sig**2))
    b = 1 / (sig * (gs.sqrt(2 * gs.pi)))
    f = b * gs.exp(-a)
    l2_norm = gs.sqrt(gs.trapz(f**2, x))
    f_sinf = f / l2_norm

    return gs.array([f_sinf])


class TestHilbertSphere(tests.conftest.TestCase):
    def setup_method(self):
        self.domain = gs.linspace(0, 1, num=50)
        self.manifold = HilbertSphere(self.domain)
        self.point_a = gaussian(self.domain, 0.2, 0.1)
        self.point_b = gaussian(self.domain, 0.5, 0.1)
        self.points = gs.squeeze(
            gs.array([gaussian(self.domain, a, 0.1) for a in gs.linspace(0.2, 0.8, 5)])
        )
        self.f = gs.sin(gs.linspace(-gs.pi, gs.pi, 50))

    def test_projection(self):
        f_proj = self.manifold.projection(self.f)
        result = self.manifold.belongs(f_proj)
        self.assertTrue(result, f"Expected True but got {result}")

    def test_belongs(self):
        result = self.manifold.belongs(self.point_a)
        self.assertTrue(result)

        result = self.manifold.belongs(self.f)
        self.assertFalse(result)

        result = gs.all(self.manifold.belongs(self.points))
        self.assertTrue(result)

    def test_to_tangent(self):
        tangent_vec = self.manifold.to_tangent(self.point_b, self.point_a)
        self.assertAllClose(gs.shape(tangent_vec), gs.shape(self.point_b))

        result = self.manifold.to_tangent(self.points, self.point_a)
        self.assertAllClose(gs.shape(result), gs.shape(self.points))

    @tests.conftest.np_and_autograd_only
    def test_is_tangent(self):
        tangent_vec = self.manifold.to_tangent(self.point_a, self.point_b)
        result = self.manifold.is_tangent(tangent_vec, self.point_b)
        self.assertTrue(result)

        tangent_vecs = self.manifold.to_tangent(self.points, self.point_a)
        result = self.manifold.is_tangent(tangent_vecs, self.point_a)
        self.assertTrue(gs.all(result))

    def test_random_point(self):
        rand_point = self.manifold.random_point()
        result = self.manifold.belongs(rand_point)
        self.assertTrue(result)

        rand_points = self.manifold.random_point(n_samples=5)
        result = gs.all(self.manifold.belongs(rand_points))
        self.assertTrue(result)


class TestHilbertSphereMetric(tests.conftest.TestCase):
    def setup_method(self):
        self.domain = gs.linspace(0, 1, num=50)
        self.f = gaussian(self.domain, 0.5, 0.1)
        self.manifold = HilbertSphere(self.domain)
        self.point_a = gaussian(self.domain, 0.2, 0.1)
        self.point_b = gaussian(self.domain, 0.5, 0.1)
        self.points = gs.squeeze(
            gs.array([gaussian(self.domain, a, 0.1) for a in gs.linspace(0.2, 0.8, 5)])
        )

    def test_inner_product(self):
        result = self.manifold.metric.inner_product(self.point_a, self.point_b)
        self.assertAllClose(gs.shape(result), (1,))
        result = self.manifold.metric.inner_product(self.points, self.point_a)
        self.assertAllClose(gs.shape(result), (gs.shape(self.points)[0],))

    def test_exp(self):
        exp = self.manifold.metric.exp(self.point_a, self.point_b)
        self.assertAllClose(gs.shape(exp), gs.shape(self.point_b))
        result = self.manifold.belongs(exp, atol=0.1)[0]
        self.assertTrue(result, f"Expected True but got {result}")
        exp = self.manifold.metric.exp(self.points, self.point_b)
        self.assertAllClose(gs.shape(exp), gs.shape(self.points))

    def test_log(self):
        log = self.manifold.metric.log(self.point_a, self.point_b)
        self.assertAllClose(gs.shape(log), gs.shape(self.point_b))
        log = self.manifold.metric.log(self.points, self.point_b)
        self.assertAllClose(gs.shape(log), gs.shape(self.points))
