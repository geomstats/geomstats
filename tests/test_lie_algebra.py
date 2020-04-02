"""Unit tests for Lie algebra."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices


class TestLieAlgebraMethods(geomstats.tests.TestCase):
    def setUp(self):
        self.n = 4
        self.dimension = int(self.n * (self.n - 1) / 2)
        self.algebra = SkewSymmetricMatrices(n=self.n)

    def test_dimension(self):
        result = self.algebra.dimension
        expected = self.dimension
        self.assertAllClose(result, expected)

    def test_basis_and_matrix_representation(self):
        n_samples = 2
        expected = gs.random.rand(n_samples * self.dimension)
        expected = gs.reshape(expected, (n_samples, self.dimension))
        mat = self.algebra.matrix_representation(expected)
        result = self.algebra.belongs(mat)
        self.assertAllClose(result, expected)

    def test_basis_and_matrix_representation(self):
        n_samples = 2
        expected = gs.random.rand(n_samples * self.dimension)
        expected = gs.reshape(expected, (n_samples, self.dimension))
        mat = self.algebra.matrix_representation(expected)
        result = self.algebra.basis_representation(mat)
        self.assertAllClose(result, expected)
