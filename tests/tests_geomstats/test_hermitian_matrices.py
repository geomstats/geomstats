"""Unit tests for the vector space of Hermitian matrices."""


import geomstats.backend as gs
from geomstats.geometry.hermitian_matrices import HermitianMatrices
from tests.conftest import Parametrizer
from tests.data.hermitian_matrices_data import HermitianMatricesTestData
from tests.geometry_test_cases import VectorSpaceTestCase


class TestHermitianMatrices(VectorSpaceTestCase, metaclass=Parametrizer):
    """Test of HermitianMatrices methods."""

    space = HermitianMatrices

    testing_data = HermitianMatricesTestData()

    def test_belongs(self, n, mat, expected):
        result = HermitianMatrices(n).belongs(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_basis(self, n, basis):
        self.assertAllClose(HermitianMatrices(n).basis, gs.array(basis))

    def test_expm(self, mat, expected):
        result = HermitianMatrices.expm(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_powerm(self, mat, power, expected):
        result = HermitianMatrices.powerm(gs.array(mat), power)
        self.assertAllClose(result, gs.array(expected))

    def test_from_vector(self, n, vec, expected):
        result = HermitianMatrices(n).from_vector(gs.array(vec), dtype=complex)
        self.assertAllClose(result, gs.array(expected))

    def test_to_vector(self, n, mat, expected):
        result = HermitianMatrices(n).to_vector(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_dim(self, n, expected_dim):
        self.assertAllClose(HermitianMatrices(n).dim, expected_dim)
