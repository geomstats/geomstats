"""Unit tests for the quotient space."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDMatrices, \
    SPDMetricBuresWasserstein
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class BuresWassersteinBundle(GeneralLinear, FiberBundle):
    def __init__(self, n):
        super(BuresWassersteinBundle, self).__init__(
            n=n, base=SPDMatrices(n), group=SpecialOrthogonal(n),
            ambient_metric=MatricesMetric(n, n))

    @staticmethod
    def riemannian_submersion(point):
        return Matrices.mul(point, Matrices.transpose(point))

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        product = Matrices.mul(
            base_point, Matrices.transpose(tangent_vec))
        return 2 * Matrices.to_symmetric(product)

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        if base_point is None:
            if fiber_point is not None:
                base_point = self.riemannian_submersion(fiber_point)
            else:
                raise ValueError('Either a point (of the total space) or a '
                                 'base point (of the base manifold) must be '
                                 'given.')
        sylvester = gs.linalg.solve_sylvester(
            base_point, base_point, tangent_vec)
        return Matrices.mul(sylvester, fiber_point)

    @staticmethod
    def lift(point):
        return gs.linalg.cholesky(point)


class TestQuotientMetric(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(0)
        n = 3
        self.bundle = BuresWassersteinBundle(n)
        self.base = self.bundle.base
        self.base_metric = SPDMetricBuresWasserstein(n)
        self.quotient_metric = QuotientMetric(self.bundle)

    def test_belongs(self):
        point = self.base.random_point()
        result = self.bundle.belongs(point)
        self.assertTrue(result)

    def test_riemannian_submersion(self):
        mat = self.bundle.random_point()
        point = self.bundle.riemannian_submersion(mat)
        result = self.bundle.belongs(point)
        self.assertTrue(result)

    def test_lift_and_riemannian_submersion(self):
        point = self.base.random_point()
        mat = self.bundle.lift(point)
        result = self.bundle.riemannian_submersion(mat)
        self.assertAllClose(result, point)

    def test_tangent_riemannian_submersion(self):
        mat = self.bundle.random_point()
        point = self.bundle.riemannian_submersion(mat)
        vec = self.bundle.random_point()
        tangent_vec = self.bundle.tangent_riemannian_submersion(vec, point)
        result = self.base.is_tangent(tangent_vec, point)
        self.assertTrue(result)

    def test_horizontal_projection(self):
        mat = self.bundle.random_point()
        vec = self.bundle.random_point()
        horizontal_vec = self.bundle.horizontal_projection(vec, mat)
        product = Matrices.mul(horizontal_vec, GeneralLinear.inverse(mat))
        is_horizontal = Matrices.is_symmetric(product)
        self.assertTrue(is_horizontal)

    def test_vertical_projection(self):
        mat = self.bundle.random_point()
        vec = self.bundle.random_point()
        vertical_vec = self.bundle.vertical_projection(vec, mat)

        result = self.bundle.tangent_riemannian_submersion(vertical_vec, mat)
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected, atol=1e-5)

    def test_horizontal_lift_and_tangent_riemannian_submersion(self):
        mat = self.bundle.random_point()
        tangent_vec = Matrices.to_symmetric(
            self.bundle.random_point())
        horizontal = self.bundle.horizontal_lift(tangent_vec, fiber_point=mat)
        result = self.bundle.tangent_riemannian_submersion(horizontal, mat)
        self.assertAllClose(result, tangent_vec)

    def test_is_horizontal(self):
        mat = self.bundle.random_point()
        tangent_vec = Matrices.to_symmetric(
            self.bundle.random_point())
        horizontal = self.bundle.horizontal_lift(tangent_vec, fiber_point=mat)
        result = self.bundle.is_horizontal(horizontal, mat)
        self.assertTrue(result)

    def test_is_vertical(self):
        mat = self.bundle.random_point()
        tangent_vec = self.bundle.random_point()
        vertical = self.bundle.vertical_projection(tangent_vec, mat)
        result = self.bundle.is_vertical(vertical, mat)
        self.assertTrue(result)

    def test_align(self):
        point = self.bundle.random_point(2)
        aligned = self.bundle.align(
            point[0], point[1], tol=1e-10)
        result = self.bundle.is_horizontal(
            point[1] - aligned, point[1], atol=1e-4)
        self.assertTrue(result)

    def test_inner_product(self):
        mat = self.bundle.random_point()
        point = self.bundle.riemannian_submersion(mat)
        tangent_vecs = Matrices.to_symmetric(
            self.bundle.random_point(2)) / 10
        result = self.quotient_metric.inner_product(
            tangent_vecs[0], tangent_vecs[1], fiber_point=mat)
        expected = self.base_metric.inner_product(
            tangent_vecs[0], tangent_vecs[1], point)
        self.assertAllClose(result, expected)

    def test_exp(self):
        mat = self.bundle.random_point()
        point = self.bundle.riemannian_submersion(mat)
        tangent_vec = Matrices.to_symmetric(
            self.bundle.random_point()) / 5

        result = self.quotient_metric.exp(tangent_vec, point)
        expected = self.base_metric.exp(tangent_vec, point)
        self.assertAllClose(result, expected)

    def test_log(self):
        mats = self.bundle.random_point(2)
        points = self.bundle.riemannian_submersion(mats)

        result = self.quotient_metric.log(points[1], points[0])
        expected = self.base_metric.log(points[1], points[0])
        self.assertAllClose(result, expected, atol=3e-4)

    def test_squared_dist(self):
        mats = self.bundle.random_point(2)
        points = self.bundle.riemannian_submersion(mats)

        result = self.quotient_metric.squared_dist(
            points[1], points[0], tol=1e-10)
        expected = self.base_metric.squared_dist(points[1], points[0])
        self.assertAllClose(result, expected)

    def test_integrability_tensor(self):
        mat = self.bundle.random_point()
        point = self.bundle.riemannian_submersion(mat)
        tangent_vec = Matrices.to_symmetric(
            self.bundle.random_point()) / 5

        self.assertRaises(
            NotImplementedError,
            lambda: self.bundle.integrability_tensor(
                tangent_vec, tangent_vec, point))
