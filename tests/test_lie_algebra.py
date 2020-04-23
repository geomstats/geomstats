"""Unit tests for Lie algebra."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices


class TestLieAlgebra(geomstats.tests.TestCase):
    def setUp(self):
        self.n = 4
        self.dim = int(self.n * (self.n - 1) / 2)
        self.algebra = SkewSymmetricMatrices(n=self.n)

    def test_dimension(self):
        result = self.algebra.dim
        expected = self.dim
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_matrix_representation_and_belongs(self):
        n_samples = 2
        point = gs.random.rand(n_samples * self.dim)
        point = gs.reshape(point, (n_samples, self.dim))
        mat = self.algebra.matrix_representation(point)
        result = self.algebra.belongs(mat).all()
        expected = True
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_basis_and_matrix_representation(self):
        n_samples = 2
        expected = gs.random.rand(n_samples * self.dim)
        expected = gs.reshape(expected, (n_samples, self.dim))
        mat = self.algebra.matrix_representation(expected)
        result = self.algebra.basis_representation(mat)
        self.assertAllClose(result, expected)
