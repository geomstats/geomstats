r"""Unit tests for the space of PSD matrices of rank k."""

import warnings

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.rank_k_psd_matrices import PSDMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices

class TestPSDMatricesRankK(geomstats.tests.TestCase):
    r"""Test of PSD Matrices Rank k methods."""

    def setUp(self):
        r"""Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

        gs.random.seed(1234)

        self.n = 3
        self.k = 2
        self.space = PSDMatrices(self.n, self.k)
        self.sym = SymmetricMatrices(self.n)

    def test_belongs(self):
        r"""Test of belongs method."""
        psd_n_k = self.space
        mat_not_psd_n_k = gs.array(
            [
                [0.27053942, -0.34773248, 0.2672531],
                [-0.34773248, 0.77543347, 0.09687998],
                [0.2672531, 0.09687998, 0.85442487],
            ]
        )
        mat_psd_n_k = gs.array([[1.0, 1.0, 3.0], [1.0, 4.0, 6.0], [3.0, 6.0, 12.0]])
        result = psd_n_k.belongs(mat_not_psd_n_k)
        expected = False
        self.assertFalse(result, expected)

        result = psd_n_k.belongs(mat_psd_n_k)
        expected = True
        self.assertTrue(result, expected)

    def test_projection_and_belongs(self):
        shape = (2, self.n, self.n)
        result = helper.test_projection_and_belongs(self.space, shape)
        for res in result:
            self.assertTrue(res)

    def test_random_and_belongs(self):
        mat = self.space.random_point(4)
        result = self.space.belongs(mat)
        self.assertTrue(gs.all(result))

    def test_is_tangent_and_to_tangent(self):
        base_point = self.space.random_point(3)
        vectors = self.sym.random_point(3)
        vectors_t = self.space.to_tangent(base_point=base_point, vector=vectors)
        vectors_t_bp0 = self.space.to_tangent(base_point=base_point[0], vector=vectors)

        result = self.space.is_tangent(base_point=base_point, vector=vectors)
        self.assertFalse(gs.all(result))
        result = self.space.is_tangent(base_point=base_point, vector=vectors_t)
        self.assertTrue(gs.all(result))


        result = self.space.is_tangent(base_point=base_point[0], vector=vectors)
        self.assertFalse(gs.all(result))
        result = self.space.is_tangent(base_point=base_point[0], vector=vectors_t_bp0)
        self.assertTrue(gs.all(result))


        result = self.space.is_tangent(base_point=base_point[0], vector=vectors[0])
        self.assertFalse(gs.all(result))
        result = self.space.is_tangent(base_point=base_point[0], vector=vectors_t_bp0[0])
        self.assertTrue(gs.all(result))
