"""Unit tests for the graphspace quotient space."""

import geomstats.tests
import geomstats.backend as gs
from geomstats.geometry.graphspace import GraphSpace, GraphSpaceMetric


class TestGraphSpace(geomstats.tests.TestCase):
    """Test of Graph Space quotient metric space."""

    def setup_method(self):
        """Set up the test."""
        gs.random.seed(1234)

        self.n = 2
        self.space = GraphSpace(n=self.n)
        self.metric = GraphSpaceMetric(n=self.n)
        self.n_samples = 10

    def test_belongs(self):
        """Test of belongs method."""
        mats = gs.array([gs.array([[3.0, -1.0], [-1.0, 3.0]]),
                         gs.array([[4.0, -6.0], [-1.0, 3.0]])])
        result = self.space.belongs(mats)
        self.assertAllClose(result, True)

        vec = gs.array([-1.0, -1.0])
        result = self.space.belongs(vec)
        self.assertAllClose(result, False)

    def test_random_point_and_belongs(self):
        """Test of random_point and belongs methods."""
        point = self.space.random_point()
        result = self.space.belongs(point)
        self.assertTrue(result)

        point = self.space.random_point(10)
        result = self.space.belongs(point)
        self.assertAllClose(result, True)

    def test_permute(self):
        """Test of permuting method."""
        graph_to_permute = gs.array([[0, 1], [2, 3]])
        permutation = gs.array([1, 0])
        expected = gs.array([[3, 2], [1, 0]])
        self.assertAllClose(expected, self.space.permute(graph_to_permute, permutation))

    def test_matchers(self):
        """Test of random_point and belongs methods."""
        set1 = gs.array([[[0, 1], [2, 3]], [[1, 0], [0, 1]]])
        set2 = gs.array([[[3, 2], [1, 0]], [[1, 0], [0, 1]]])
        d1 = self.metric.dist(set1, set2, matcher='FAQ')
        d2 = self.metric.dist(set1, set2, matcher='ID')
        self.assertTrue(d1[0] < d2[0])
        self.assertTrue(d1[1] == d2[1])
