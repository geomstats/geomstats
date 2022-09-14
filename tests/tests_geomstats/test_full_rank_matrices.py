"""Unit tests for full rank matrices."""

import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.full_rank_matrices_data import FullRankMatricesTestData
from tests.geometry_test_cases import OpenSetTestCase


class TestFullRankMatrices(OpenSetTestCase, metaclass=Parametrizer):

    testing_data = FullRankMatricesTestData()

    def test_belongs(self, m, n, mat, expected):
        self.assertAllClose(self.Space(m, n).belongs(gs.array(mat)), gs.array(expected))
