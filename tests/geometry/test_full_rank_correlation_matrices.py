"""Unit tests for the manifold of matrices."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.full_rank_correlation_matrices import \
    CorrelationMatricesBundle, FullRankCorrelationAffineQuotientMetric
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SymmetricMatrices


class TestFullRankCorrelationMatrices(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(12)
        n = 3
        self.n = n
        bundle = CorrelationMatricesBundle(n)
        self.bundle = bundle
        self.base = bundle.base
        self.quotient_metric = FullRankCorrelationAffineQuotientMetric(n)

    def test_belongs(self):
        point = self.base.random_point()
        result = self.base.belongs(point)
        self.assertTrue(result)

    def test_riemannian_submersion(self):
        mat = self.bundle.random_point()
        point = self.bundle.riemannian_submersion(mat)
        result = self.base.belongs(point)
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

    def test_vertical_projection(self):
        mat = self.bundle.random_point(2)
        vec = SymmetricMatrices(self.n).random_point(2)
        tangent_vec = self.bundle.to_tangent(vec, mat)
        vertical = self.bundle.vertical_projection(tangent_vec, mat)
        result = self.bundle.tangent_riemannian_submersion(
            vertical, mat)

        expected = gs.zeros_like(vec)
        self.assertAllClose(result, expected)

    def test_horizontal_projection(self):
        mat = self.bundle.random_point()
        vec = self.bundle.random_point()
        horizontal_vec = self.bundle.horizontal_projection(vec, mat)
        inverse = GeneralLinear.inverse(mat)
        product_1 = Matrices.mul(horizontal_vec, inverse)
        product_2 = Matrices.mul(inverse, horizontal_vec)
        is_horizontal = self.base.is_tangent(
            product_1 + product_2, mat, atol=gs.atol * 10)
        self.assertTrue(is_horizontal)

    def test_horizontal_lift_and_tangent_riemannian_submersion(self):
        mat = self.base.random_point()
        vec = self.base.random_point()
        tangent_vec = self.base.to_tangent(vec, mat)
        horizontal = self.bundle.horizontal_lift(tangent_vec, base_point=mat)
        result = self.bundle.tangent_riemannian_submersion(horizontal, mat)
        self.assertAllClose(result, tangent_vec)

    def test_is_horizontal(self):
        mat = self.base.random_point()
        vec = self.base.random_point()
        tangent_vec = self.base.to_tangent(vec, mat)
        horizontal = self.bundle.horizontal_lift(tangent_vec, mat)
        result = self.bundle.is_horizontal(horizontal, mat)
        self.assertTrue(result)

    def test_is_vertical(self):
        mat = self.bundle.random_point()
        tangent_vec = self.bundle.random_point()
        vertical = self.bundle.vertical_projection(tangent_vec, mat)
        result = self.bundle.is_vertical(vertical, mat)
        self.assertTrue(result)

    def test_inner_product(self):

        def inner_prod(tangent_vec_a, tangent_vec_b, base_point):
            affine_part = self.bundle.ambient_metric.inner_product(
                tangent_vec_a, tangent_vec_b, base_point)
            n = tangent_vec_b.shape[-1]

            inverse_base_point = GeneralLinear.inverse(base_point)
            operator = gs.eye(n) + base_point * inverse_base_point
            inverse_operator = GeneralLinear.inverse(operator)

            diagonal_a = gs.einsum(
                '...ij,...ji->...i', inverse_base_point, tangent_vec_a)
            diagonal_b = gs.einsum(
                '...ij,...ji->...i', inverse_base_point, tangent_vec_b)
            aux = gs.einsum('...i,...j->...ij', diagonal_a, diagonal_b)
            other_part = 2 * Matrices.frobenius_product(aux, inverse_operator)
            return affine_part - other_part

        mat = self.base.random_point()
        vecs = self.bundle.random_point(2)
        tangent_vecs = self.base.to_tangent(vecs, mat)
        result = self.quotient_metric.inner_product(
            tangent_vecs[0], tangent_vecs[1], base_point=mat)
        expected = inner_prod(
            tangent_vecs[0], tangent_vecs[1], base_point=mat)
        self.assertAllClose(result, expected)

    def test_exp_and_belongs(self):
        point = self.base.random_point()
        vec = self.bundle.random_point()
        tangent_vec = self.base.to_tangent(vec, point)

        exp = self.quotient_metric.exp(tangent_vec, point)
        result = self.base.belongs(exp)
        self.assertTrue(result)

    def test_align(self):
        point = self.bundle.random_point(2)
        aligned = self.bundle.align(
            point[0], point[1], tol=1e-10)
        log = self.bundle.ambient_metric.log(aligned, point[1])
        result = self.bundle.is_horizontal(
            log, point[1], atol=gs.atol * 100)
        self.assertTrue(result)

    def test_exp_and_log(self):
        mats = self.bundle.random_point(2)
        points = self.bundle.riemannian_submersion(mats)

        log = self.quotient_metric.log(points[1], points[0])
        result = self.quotient_metric.exp(log, points[0])
        self.assertAllClose(result, points[1], atol=gs.atol * 100)
