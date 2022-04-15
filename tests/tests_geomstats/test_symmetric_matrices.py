"""Unit tests for the vector space of symmetric matrices."""


import geomstats.backend as gs
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.conftest import Parametrizer
from tests.data.symmetric_matrices_data import SymmetricMatricesTestData
from tests.geometry_test_cases import VectorSpaceTestCase


class TestSymmetricMatrices(VectorSpaceTestCase, metaclass=Parametrizer):
    """Test of SymmetricMatrices methods."""

    space = SymmetricMatrices

    testing_data = SymmetricMatricesTestData()

    def test_belongs(self, n, mat, expected):
        result = SymmetricMatrices(n).belongs(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_basis(self, n, basis):
        self.assertAllClose(SymmetricMatrices(n).basis, gs.array(basis))

    def test_expm(self, mat, expected):
        result = SymmetricMatrices.expm(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_powerm(self, mat, power, expected):
        result = SymmetricMatrices.powerm(gs.array(mat), power)
        self.assertAllClose(result, gs.array(expected))

    def test_from_vector(self, n, vec, expected):
        result = SymmetricMatrices(n).from_vector(gs.array(vec))
        self.assertAllClose(result, gs.array(expected))

    def test_to_vector(self, n, mat, expected):
        result = SymmetricMatrices(n).to_vector(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_dim(self, n, expected_dim):
        self.assertAllClose(SymmetricMatrices(n).dim, expected_dim)
