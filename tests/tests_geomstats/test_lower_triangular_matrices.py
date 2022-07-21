"""Unit tests for the vector space of lower triangular matrices."""

import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.lower_triangular_matrices_data import LowerTriangularMatricesTestData
from tests.geometry_test_cases import VectorSpaceTestCase


class TestLowerTriangularMatrices(VectorSpaceTestCase, metaclass=Parametrizer):
    """Test of LowerTriangularMatrices methods."""

    testing_data = LowerTriangularMatricesTestData()
    space = testing_data.space

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(self.space(n).belongs(gs.array(mat)), gs.array(expected))

    def test_random_point_and_belongs(self, n, n_points):
        space_n = self.space(n)
        self.assertAllClose(
            gs.all(space_n.belongs(space_n.random_point(n_points))), True
        )

    def test_to_vector(self, n, mat, expected):
        self.assertAllClose(self.space(n).to_vector(gs.array(mat)), gs.array(expected))

    def test_get_basis(self, n, expected):
        self.assertAllClose(self.space(n).basis, gs.array(expected))

    def test_projection(self, n, point, expected):
        self.assertAllClose(
            self.space(n).projection(gs.array(point)), gs.array(expected)
        )
