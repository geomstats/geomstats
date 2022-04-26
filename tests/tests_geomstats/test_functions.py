"""Unit tests for the functions manifolds."""

import numpy as np

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.functions import HilbertSphere


class TestFunctionsSinf(geomstats.tests.TestCase):
    def setup_method(self):
        self.domain = gs.linspace(-gs.pi, gs.pi, num=50)
        self.funcs = lambda a: np.sin(a * self.domain).reshape(1, 50)
        self.f = self.funcs(1)
        self.manifold = HilbertSphere(self.domain)

    @geomstats.tests.np_and_autograd_only
    def test_manifold(self):
        proj_f = self.manifold.projection(self.f)
        result = self.manifold.belongs(proj_f)
        self.assertTrue(result, "Expected True but got %s" % result)

    @geomstats.tests.np_and_autograd_only
    def test_metric(self):
        num_funcs = 10
        point_a = self.funcs(1)
        point_b = self.funcs(2)
        result = self.manifold.metric.inner_product(point_a, point_b)
        self.assertAllClose(gs.shape(result), (1,))

        proj_points = gs.array(
            [
                self.manifold.projection(self.funcs(a))
                for a in gs.linspace(1, 5, num=num_funcs)
            ]
        ).squeeze()
        result = self.manifold.metric.inner_product(proj_points, point_b)
        self.assertAllClose(len(result), len(proj_points))

        tangent_vec = self.manifold.metric.log(point_b, point_a)
        point_b_exp = self.manifold.metric.exp(tangent_vec, point_a)
        self.assertAllClose(gs.shape(point_b_exp), gs.shape(point_a))

        tangent_vecs = self.manifold.metric.log(proj_points, point_a)
        proj_points_exp = self.manifold.metric.exp(tangent_vecs, point_a)
        self.assertAllClose(gs.shape(proj_points_exp), gs.shape(proj_points))
