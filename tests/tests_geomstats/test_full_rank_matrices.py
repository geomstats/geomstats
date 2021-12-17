"""Unit tests for full rank matrices."""

import warnings

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.full_rank_matrices import FullRankMatrices


class TestFullRankMatrices(geomstats.tests.TestCase):
    """Test of Full Rank Matrices methods."""

    def setup_method(self):
        """Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

        gs.random.seed(1234)

        self.m = 3
        self.n = 2
        self.space = FullRankMatrices(self.m, self.n)

    def test_belongs(self):
        """Test of belongs method."""
        fr = self.space
        mat_fr = gs.array(
            [
                [-1.6473486, -1.18240309],
                [0.1944016, 0.18169231],
                [-1.13933855, -0.64971248],
            ]
        )
        mat_not_fr = gs.array([[1.0, -1.0], [1.0, -1.0], [0.0, 0.0]])
        result = fr.belongs(mat_fr)
        self.assertTrue(result)
        result = fr.belongs(mat_not_fr)
        self.assertFalse(result)

    def test_projection_and_belongs(self):
        """Test of projection method."""
        shape = (2, self.m, self.n)
        result = helper.test_projection_and_belongs(self.space, shape, atol=gs.atol)
        for res in result:
            self.assertTrue(res)

    def test_random_and_belongs(self):
        """Test of random point sampling method."""
        mat = self.space.random_point(5)
        result = self.space.belongs(mat)
        self.assertTrue(gs.all(result))
