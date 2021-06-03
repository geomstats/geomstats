"""Unit tests for the manifold of matrices."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.full_rank_correlation_matrices import \
    CorrelationMatricesBundle, FullRankCorrelationMatrices,\
    FullRankCorrelationAffineQuotientMetric
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices, SymmetricMatrices


class TestCorrelationMatrices(geomstats.tests.TestCase):
    def setUp(self):
        n = 3
        self.n = n
        self.bundle = CorrelationMatricesBundle(n)
        self.base = FullRankCorrelationMatrices(n)

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
        mat = self.bundle.random_point()
        vec = SymmetricMatrices(self.n).random_point()
        tangent_vec = self.bundle.to_tangent(vec, mat)
        vertical = self.bundle.vertical_projection(tangent_vec, mat)
        result = self.bundle.tangent_riemannian_submersion(
            vertical, mat)

        expected = gs.zeros_like(vec)
        self.assertAllClose(result, expected)
