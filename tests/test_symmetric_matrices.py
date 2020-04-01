"""Unit tests for the vector space of symmetric matrices."""

import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.symmetric_matrices import SymmetricMatrices



class TestSymmetricMatricesMethods(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

        gs.random.seed(1234)

        self.n = 3

    def test_belongs(self):
        Sym_n = SymmetricMatrices(self.n)
        mat_sym = gs.array([[1., 2., 3.],
                            [2., 4., 5.],
                            [3., 5., 6.]])
        mat_not_sym = gs.array([[1., 0., 3.],
                                [2., 4., 5.],
                                [3., 5., 6.]])
        result = Sym_n.belongs(mat_sym) and not Sym_n.belongs(mat_not_sym)
        expected = True
        self.assertAllClose(result, expected)
