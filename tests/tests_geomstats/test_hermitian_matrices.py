"""Unit tests for the vector space of Hermitian matrices."""


import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.hermitian_matrices import HermitianMatrices
from tests.conftest import Parametrizer
from tests.data.hermitian_matrices_data import HermitianMatricesTestData
from tests.geometry_test_cases import VectorSpaceTestCase

CDTYPE = gs.get_default_cdtype()


class TestHermitianMatrices(VectorSpaceTestCase, metaclass=Parametrizer):
    """Test of HermitianMatrices methods."""

    testing_data = HermitianMatricesTestData()
    Space = testing_data.Space

    def test_belongs(self, n, mat, expected):
        result = self.Space(n).belongs(gs.array(mat, dtype=CDTYPE))
        self.assertAllClose(result, gs.array(expected))

    def test_basis(self, n, basis):
        self.assertAllClose(self.Space(n).basis, gs.array(basis, dtype=CDTYPE))

    def test_expm(self, mat, expected):
        result = HermitianMatrices.expm(gs.array(mat, dtype=CDTYPE))
        self.assertAllClose(result, gs.array(expected, dtype=CDTYPE))

    @tests.conftest.np_autograd_and_torch_only
    def test_powerm(self, mat, power, expected):
        result = HermitianMatrices.powerm(gs.array(mat, dtype=CDTYPE), power)
        self.assertAllClose(result, gs.array(expected, dtype=CDTYPE))

    def test_from_vector(self, n, vec, expected):
        result = self.Space(n).from_vector(gs.array(vec, dtype=CDTYPE))
        self.assertAllClose(result, gs.array(expected, dtype=CDTYPE))

    def test_to_vector(self, n, mat, expected):
        result = self.Space(n).to_vector(gs.array(mat, dtype=CDTYPE))
        self.assertAllClose(result, gs.array(expected, dtype=CDTYPE))

    def test_dim(self, n, expected_dim):
        self.assertAllClose(self.Space(n).dim, expected_dim)
