r"""Unit tests for the space of PSD matrices of rank k."""

import warnings

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.rank_k_psd_matrices import PSDMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


class TestPSDMatricesRankK(geomstats.tests.TestCase):
    r"""Test of PSD Matrices Rank k methods."""

    def setup_method(self):
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
                [0.8369314, -0.7342977, 1.0402943],
                [0.04035992, -0.7218659, 1.0794858],
                [0.9032698, -0.73601735, -0.36105633],
            ]
        )
        mat_psd_n_k = gs.array([[1.0, 1.0, 0], [1.0, 4.0, 0], [0, 0, 0]])
        result = psd_n_k.belongs(mat_not_psd_n_k)
        self.assertFalse(result)

        result = psd_n_k.belongs(mat_psd_n_k)
        self.assertTrue(result)

    def test_projection_and_belongs(self):
        r"""Test the projection and the belongs methods."""
        points = self.sym.random_point(3)
        proj_points = self.space.projection(points)
        result = self.space.belongs(proj_points)
        self.assertTrue(gs.all(result))

    def test_random_and_belongs(self):
        r"""Test the random and the belongs methods."""
        mat = self.space.random_point(4)
        result = self.space.belongs(mat)
        self.assertTrue(gs.all(result))

    def test_is_tangent_and_to_tangent(self):
        r"""Test the tangent functions."""
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
        result = self.space.is_tangent(
            base_point=base_point[0], vector=vectors_t_bp0[0]
        )
        self.assertTrue(gs.all(result))
