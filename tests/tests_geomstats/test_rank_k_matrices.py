"""Unit tests for full rank matrices."""

import warnings
import tests.helper as helper
import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.rank_k_matrices import RankKMatrices


class TestRankMatrices(geomstats.tests.TestCase):
    """Test of Full Rank Matrices methods."""

    def setUp(self):
        """Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

        gs.random.seed(1234)

        self.m = 4
        self.n = 3
        self.rank = 2
        self.space = RankKMatrices(m=self.m, n=self.n, k=self.rank)

    def test_belongs(self):
        """Test of belongs method."""
        fr = self.space
        mat_full_r = gs.array(
            [
                [-0.01159978, 0.11877343, 1.60720463],
                [-0.04523795, -0.24663408, -0.28083758],
                [-1.01415994, -0.16618452, -0.26646112],
                [2.16092311, 0.62531989, 0.16770327],
            ]
        )
        mat_rank_correct = gs.array([[1, 2, 3], [2, 4, 6], [6, 12, 18], [7, 3, 1]])
        result = fr.belongs(mat_full_r)
        expected = False
        self.assertAllClose(result, expected)

        result = fr.belongs(mat_rank_correct)
        expected = True
        self.assertAllClose(result, expected)

    def test_projection_and_belongs(self):
        """Test of projection method."""
        shape = (2, self.m, self.n)
        result = helper.test_projection_and_belongs(self.space, shape)
        for res in result:
            self.assertTrue(res)

    def test_random_and_belongs(self):
        """Test of random point sampling method."""
        mat = self.space.random_point()
        result = self.space.belongs(mat)
        self.assertTrue(result)
