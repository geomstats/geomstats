"""Unit tests for rank k matrices."""

import warnings
import tests.helper as helper
import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.rank_k_matrices import RankKMatrices


class TestRankKMatrices(geomstats.tests.TestCase):
    """Test of Rank k Matrices methods."""

    def setUp(self):
        """Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

        gs.random.seed(1234)

        self.m = 7
        self.n = 5
        self.rank = 2
        self.space = RankKMatrices(m=self.m, n=self.n, k=self.rank)

    def test_belongs(self):
        """Test of belongs method."""
        fr = self.space
        mat_rk = gs.array([[ 1.44859394,  0.1281937 , -0.3425299 , -0.31322621, -0.19949473],
            [ 0.17748325, -1.61277241, -0.54447072,  1.51380882,  0.28449503],
            [-0.40216044, -0.40998888, -0.0204358 ,  0.44381745,  0.126411  ],
            [-1.33446687, -0.29198054,  0.26188719,  0.45428884,  0.21676546],
            [-0.3131184 , -1.78270634, -0.46750458,  1.74048123,  0.37606041],
            [-0.49552776, -1.78361897, -0.41967317,  1.76540697,  0.3982919 ],
            [ 1.70266739,  0.57588033, -0.2714016 , -0.77344581, -0.31514951]])
        mat_not_rk = gs.array([[-2.47656966,  0.81578582, -0.27115525, -1.52809339,  0.57906149],
            [-1.27575259, -0.47427609,  0.03346678, -1.62001201,  0.47961903],
            [ 1.65259633, -0.15066283,  0.14864467, -0.43929151, -0.78737939],
            [ 0.32529754,  0.25502012, -0.02821748, -0.33302278, -0.32519664],
            [-0.40183341,  0.02389871, -0.68031397, -1.52017816, -1.14055146],
            [ 0.67272994,  0.79642649,  1.07273908,  0.43888775,  1.27266412],
            [ 1.03067103, -1.38305793,  1.08433112,  0.4527683 ,  1.31748468]])
        result = fr.belongs(mat_not_rk)
        expected = False
        self.assertAllClose(result, expected)

        result = fr.belongs(mat_rk)
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
