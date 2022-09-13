"""Unit tests for the skew symmetric matrices."""

import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.skew_symmetric_matrices_data import SkewSymmetricMatricesTestData
from tests.geometry_test_cases import MatrixLieAlgebraTestCase


class TestSkewSymmetricMatrices(MatrixLieAlgebraTestCase, metaclass=Parametrizer):

    testing_data = SkewSymmetricMatricesTestData()

    def test_belongs(self, n, mat, expected):
        skew = self.Space(n)
        self.assertAllClose(skew.belongs(gs.array(mat)), gs.array(expected))

    def test_baker_campbell_hausdorff(self, n, matrix_a, matrix_b, order, expected):
        skew = self.Space(n)
        result = skew.baker_campbell_hausdorff(
            gs.array(matrix_a), gs.array(matrix_b), order=order
        )
        self.assertAllClose(result, gs.array(expected))
