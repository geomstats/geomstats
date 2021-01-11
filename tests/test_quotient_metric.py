"""Unit tests for the quotient space."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import MatricesMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDMatrices, \
    SPDMetricBuresWasserstein
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class TestQuotientMetric(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(0)
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
        self.bundle.lift = gs.linalg.cholesky

    def test_belongs(self):
        point = self.base.random_uniform()
        result = self.bundle.belongs(point)
        self.assertTrue(result)

    def test_submersion(self):
        mat = self.bundle.total_space.random_uniform()
        point = self.bundle.submersion(mat)
        result = self.bundle.belongs(point)
        self.assertTrue(result)

    def test_lift_and_submersion(self):
        point = self.base.random_uniform()
        mat = self.bundle.lift(point)
        result = self.bundle.submersion(mat)
        self.assertAllClose(result, point)

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
        self.assertAllClose(result, expected, atol=1e-5)

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
        tangent_vec = self.bundle.total_space.random_uniform()
        vertical = self.bundle.vertical_projection(tangent_vec, mat)
        result = self.bundle.is_vertical(vertical, mat)
        self.assertTrue(result)

    def test_align(self):
        point = self.bundle.total_space.random_uniform(2)
        aligned = self.bundle.align(
            point[0], point[1], tol=1e-10)
        result = self.bundle.is_horizontal(
            point[1] - aligned, point[1], atol=1e-5)
        self.assertTrue(result)

    def test_inner_product(self):
        mat = self.bundle.total_space.random_uniform()
        point = self.bundle.submersion(mat)
        tangent_vecs = GeneralLinear.to_symmetric(
            self.bundle.total_space.random_uniform(2)) / 10
        result = self.quotient_metric.inner_product(
            tangent_vecs[0], tangent_vecs[1], point=mat)
        expected = self.base_metric.inner_product(
            tangent_vecs[0], tangent_vecs[1], point)
        self.assertAllClose(result, expected)

    def test_exp(self):
        mat = self.bundle.total_space.random_uniform()
        point = self.bundle.submersion(mat)
        tangent_vec = GeneralLinear.to_symmetric(
            self.bundle.total_space.random_uniform()) / 5

        result = self.quotient_metric.exp(tangent_vec, point)
        expected = self.base_metric.exp(tangent_vec, point)
        self.assertAllClose(result, expected)

    def test_log(self):
        mats = self.bundle.total_space.random_uniform(2)
        points = self.bundle.submersion(mats)

        result = self.quotient_metric.log(points[1], points[0], tol=1e-10)
        expected = self.base_metric.log(points[1], points[0])
        self.assertAllClose(result, expected, atol=3e-4)

    def test_squared_dist(self):
        mats = self.bundle.total_space.random_uniform(2)
        points = self.bundle.submersion(mats)

        result = self.quotient_metric.squared_dist(
            points[1], points[0], tol=1e-10)
        expected = self.base_metric.squared_dist(points[1], points[0])
        self.assertAllClose(result, expected, atol=1e-5)
