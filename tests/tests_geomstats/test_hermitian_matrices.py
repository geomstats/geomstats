"""Unit tests for the vector space of Hermitian matrices."""


from geomstats.geometry.hermitian_matrices import HermitianMatrices
from tests.conftest import Parametrizer
from tests.data.hermitian_matrices_data import HermitianMatricesTestData
from tests.geometry_test_cases import VectorSpaceTestCase


class TestHermitianMatrices(VectorSpaceTestCase, metaclass=Parametrizer):
    """Test of HermitianMatrices methods."""

    testing_data = HermitianMatricesTestData()
    Space = testing_data.Space

    def test_belongs(self, n, mat, expected):
        result = self.Space(n).belongs(mat)
        self.assertAllClose(result, expected)

    def test_basis(self, n, basis):
        self.assertAllClose(self.Space(n).basis, basis)

    def test_expm(self, mat, expected):
        result = HermitianMatrices.expm(mat)
        self.assertAllClose(result, expected)

    def test_powerm(self, mat, power, expected):
        result = HermitianMatrices.powerm(mat, power)
        self.assertAllClose(result, expected)

    def test_from_vector(self, n, vec, expected):
        result = self.Space(n).from_vector(vec)
        self.assertAllClose(result, expected)

    def test_to_vector(self, n, mat, expected):
        result = self.Space(n).to_vector(mat)
        self.assertAllClose(result, expected)

    def test_dim(self, n, expected_dim):
        self.assertAllClose(self.Space(n).dim, expected_dim)
