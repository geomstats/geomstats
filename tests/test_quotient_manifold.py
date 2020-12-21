"""Unit tests for the quotient space."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import MatricesMetric
from geomstats.geometry.spd_matrices import SPDMatrices, \
    SPDMetricBuresWasserstein
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.quotient_manifold import FiberBundle, QuotientMetric


class TestPreShapeSpace(geomstats.tests.TestCase):
    def setUp(self):
        n = 3
        self.base = SPDMatrices(n)
        self.base_metric = SPDMetricBuresWasserstein(n)
        self.group = SpecialOrthogonal(n)
        self.bundle = FiberBundle(
            GeneralLinear(n), base=self.base, group=self.group)
        self.quotient_metric = QuotientMetric(
            self.bundle, ambient_metric=MatricesMetric(n, n))

        def submersion(point):
            return GeneralLinear.mul(point, GeneralLinear.transpose(point))

        def tangent_submersion(tangent_vec, base_point):
            product = GeneralLinear.mul(
                base_point, GeneralLinear.transpose(tangent_vec))
            return 2 * GeneralLinear.to_symmetric(product)

        def horizontal_lift(tangent_vec, point, base_point=None):
            if base_point is None:
                base_point = submersion(point)
            sylvester = gs.linalg.solve_sylvester(
                base_point, base_point, tangent_vec)
            return GeneralLinear.mul(sylvester, point)

        self.bundle.submersion = submersion
        self.bundle.tangent_submersion = tangent_submersion
        self.bundle.horizontal_lift = horizontal_lift

    def test_belongs(self):
        point = self.base.random_uniform()
        result = self.bundle.belongs(point)
        self.assertTrue(result)

    def test_submersion(self):
        mat = self.bundle.total_space.random_uniform()
        point = self.bundle.submersion(mat)
        result = self.bundle.belongs(point)
        self.assertTrue(result)

    def test_tangent_submersion(self):
        mat = self.bundle.total_space.random_uniform()
        point = self.bundle.submersion(mat)
        vec = self.bundle.total_space.random_uniform()
        tangent_vec = self.bundle.tangent_submersion(vec, point)
        result = self.base.is_tangent(tangent_vec, point)
        self.assertTrue(result)

    def test_horizontal_projection(self):
        mat = self.bundle.total_space.random_uniform()
        vec = self.bundle.total_space.random_uniform()
        horizontal_vec = self.bundle.horizontal_projection(vec, mat)
        product = GeneralLinear.mul(horizontal_vec, GeneralLinear.inverse(mat))
        is_horizontal = GeneralLinear.is_symmetric(product)
        self.assertTrue(is_horizontal)

    def test_vertical_projection(self):
        mat = self.bundle.total_space.random_uniform()
        vec = self.bundle.total_space.random_uniform()
        vertical_vec = self.bundle.vertical_projection(vec, mat)

        result = self.bundle.tangent_submersion(vertical_vec, mat)
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected)

    def test_horizontal_lift_and_tangent_submersion(self):
        mat = self.bundle.total_space.random_uniform()
        tangent_vec = GeneralLinear.to_symmetric(
            self.bundle.total_space.random_uniform())
        horizontal = self.bundle.horizontal_lift(tangent_vec, mat)
        result = self.bundle.tangent_submersion(horizontal, mat)
        self.assertAllClose(result, tangent_vec)

    def test_is_horizontal(self):
        mat = self.bundle.total_space.random_uniform()
        tangent_vec = GeneralLinear.to_symmetric(
            self.bundle.total_space.random_uniform())
        horizontal = self.bundle.horizontal_lift(tangent_vec, mat)
        result = self.bundle.is_horizontal(horizontal, mat)
        self.assertTrue(result)

    def test_is_vertical(self):
        mat = self.bundle.total_space.random_uniform()
        tangent_vec = GeneralLinear.to_symmetric(
            self.bundle.total_space.random_uniform())
        horizontal = self.bundle.vertical_projection(tangent_vec, mat)
        result = self.bundle.is_vertical(horizontal, mat)
        self.assertTrue(result)

    def test_squared_dist(self):
        point = self.base.random_uniform(2)
        result = self.quotient_metric.squared_dist(point[0], point[1])
        expected = self.base_metric.squared_dist(point[0], point[1])
        self.assertAllClose(result, expected)
