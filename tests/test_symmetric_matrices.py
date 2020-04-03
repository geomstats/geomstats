"""Unit tests for the vector space of symmetric matrices."""

import math
import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


class TestSymmetricMatricesMethods(geomstats.tests.TestCase):
    """Test of SymmetricMatrices methods."""

    def setUp(self):
        """Set up the test."""
        warnings.simplefilter('ignore', category=ImportWarning)

        gs.random.seed(1234)

        self.n = 3

    def test_belongs(self):
        """Test of belongs method."""
        sym_n = SymmetricMatrices(self.n)
        mat_sym = gs.array([[1., 2., 3.],
                            [2., 4., 5.],
                            [3., 5., 6.]])
        mat_not_sym = gs.array([[1., 0., 3.],
                                [2., 4., 5.],
                                [3., 5., 6.]])
        result = sym_n.belongs(mat_sym) and not sym_n.belongs(mat_not_sym)
        expected = True
        self.assertAllClose(result, expected)

    def test_expm(self):
        """Test of expm method."""
        sym_n = SymmetricMatrices(self.n)
        v = gs.array([[[0., 1., 0.],
                       [1., 0., 0.],
                       [0., 0., 1.]]])
        result = sym_n.expm(v)
        c = math.cosh(1)
        s = math.sinh(1)
        e = math.exp(1)
        expected = gs.array([[[c, s, 0.],
                              [s, c, 0.],
                              [0., 0., e]]])
        self.assertAllClose(result, expected)
