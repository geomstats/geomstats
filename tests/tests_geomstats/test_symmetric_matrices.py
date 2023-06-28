"""Unit tests for the vector space of symmetric matrices."""

from tests.conftest import Parametrizer
from tests.data.symmetric_matrices_data import SymmetricMatricesTestData
from tests.geometry_test_cases import VectorSpaceTestCase


class TestSymmetricMatrices(VectorSpaceTestCase, metaclass=Parametrizer):
    """Test of SymmetricMatrices methods."""

    testing_data = SymmetricMatricesTestData()

    def test_belongs(self, n, mat, expected):
        result = self.Space(n).belongs(mat)
        self.assertAllClose(result, expected)

    def test_basis(self, n, basis):
        self.assertAllClose(self.Space(n).basis, basis)

    def test_expm(self, mat, expected):
        result = self.Space.expm(mat)
        self.assertAllClose(result, expected)

    def test_powerm(self, mat, power, expected):
        result = self.Space.powerm(mat, power)
        self.assertAllClose(result, expected)

    def test_from_vector(self, n, vec, expected):
        result = self.Space(n).from_vector(vec)
        self.assertAllClose(result, expected)

    def test_to_vector(self, n, mat, expected):
        result = self.Space(n).to_vector(mat)
        self.assertAllClose(result, expected)

    def test_dim(self, n, expected_dim):
        self.assertAllClose(self.Space(n).dim, expected_dim)
