"""Unit tests for the functions manifolds."""

import math
import warnings

import numpy as np

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.functions import SinfSpace


class TestFunctionsSinf(geomstats.tests.TestCase):
    def setup_method(self):
        domain = gs.linspace(-math.pi, math.pi)
        self.f = gs.sin(domain)
        self.f_sinf = self.f / np.trapz(self.f, domain)
        self.manifold = SinfSpace(domain)
        self.funcs = lambda a: np.sin(a * domain).reshape(1, num_samples)

    def test_manifold(self):
        result = self.manifold.belongs(self.f_sinf)
        self.assertTrue(result)

        proj_f = self.manifold.projection(f)
        result = self.manifold.belongs(proj_f)
        self.assertTrue(result)

    def test_metric(self):
        point_a = self.funcs(1)
        point_b = self.funcs(2)
        result = self.manifold.metric.inner_product(point_a, point_b)
        self.assertAllClose(gs.shape(result), ())

        proj_points = gs.array(
            [manifold.projection(self.funcs(a)) for a in gs.linspace(1, 5, num=10)]
        ).squeeze()
        result = manifold.metric.inner_product(proj_points, point_b)
        self.assertAllClose(gs.shape(result), (len(proj_points),))

        tangent_vec = manifold.metric.log(point_b, point_a)
        self.assertTrue(self.manifold.is_tangent(tangent_vec))
        point_b_exp = manifold.metric.exp(tangent_vec, point_a)
        self.assertTrue(self.manifold.belongs(point_b_exp))

        tangent_vecs = manifold.metric.log(proj_points, point_a)
        self.assertTrue(gs.all(self.manifold.is_tangent(tangent_vecs)))
        proj_points_exp = manifold.metric.exp(tangent_vecs, point_a)
        self.assertTrue(self.manifold.belongs(proj_points_exp))
