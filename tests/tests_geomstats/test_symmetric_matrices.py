"""Unit tests for the vector space of symmetric matrices."""


import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.symmetric_matrices_data import SymmetricMatricesTestData
from tests.geometry_test_cases import VectorSpaceTestCase


class TestSymmetricMatrices(VectorSpaceTestCase, metaclass=Parametrizer):
    """Test of SymmetricMatrices methods."""

    testing_data = SymmetricMatricesTestData()

    def test_belongs(self, n, mat, expected):
        result = self.Space(n).belongs(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_basis(self, n, basis):
        self.assertAllClose(self.Space(n).basis, gs.array(basis))

    def test_expm(self, mat, expected):
        result = self.Space.expm(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_powerm(self, mat, power, expected):
        result = self.Space.powerm(gs.array(mat), power)
        self.assertAllClose(result, gs.array(expected))

    def test_from_vector(self, n, vec, expected):
        result = self.Space(n).from_vector(gs.array(vec))
        self.assertAllClose(result, gs.array(expected))

    def test_to_vector(self, n, mat, expected):
        result = self.Space(n).to_vector(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_dim(self, n, expected_dim):
        self.assertAllClose(self.Space(n).dim, expected_dim)
