"""Unit tests for the vector space of symmetric matrices."""


import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.symmetric_matrices_data import SymmetricMatricesTestData
from tests.geometry_test_cases import VectorSpaceTestCase


class TestSymmetricMatrices(VectorSpaceTestCase, metaclass=Parametrizer):
    """Test of SymmetricMatrices methods."""

    testing_data = SymmetricMatricesTestData()
    space = testing_data.space

    def test_belongs(self, n, mat, expected):
        result = self.space(n).belongs(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_basis(self, n, basis):
        self.assertAllClose(self.space(n).basis, gs.array(basis))

    def test_expm(self, mat, expected):
        result = self.space.expm(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_powerm(self, mat, power, expected):
        result = self.space.powerm(gs.array(mat), power)
        self.assertAllClose(result, gs.array(expected))

    def test_from_vector(self, n, vec, expected):
        result = self.space(n).from_vector(gs.array(vec))
        self.assertAllClose(result, gs.array(expected))

    def test_to_vector(self, n, mat, expected):
        result = self.space(n).to_vector(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_dim(self, n, expected_dim):
        self.assertAllClose(self.space(n).dim, expected_dim)
