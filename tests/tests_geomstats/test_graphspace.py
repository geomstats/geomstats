"""Unit tests for the graphspace quotient space."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.graphspace import GraphSpace, GraphSpaceMetric


class TestGraphSpace(geomstats.tests.TestCase):
    """Test of Graph Space quotient metric space."""

    def setup_method(self):
        """Set up the test."""
        gs.random.seed(1234)

        self.n = 2
        self.space = GraphSpace(nodes=self.n)
        self.metric = GraphSpaceMetric(nodes=self.n)
        self.n_samples = 10

    def test_belongs(self):
        """Test of belongs method."""
        mats = gs.array([[[3.0, -1.0], [-1.0, 3.0]], [[4.0, -6.0], [-1.0, 3.0]]])
        result = self.space.belongs(mats)
        self.assertTrue(gs.all(result))

        vec = gs.array([-1.0, -1.0])
        result = self.space.belongs(vec)
        self.assertFalse(gs.all(result))

    def test_random_point_and_belongs(self):
        """Test of random_point and belongs methods."""
        point = self.space.random_point()
        result = self.space.belongs(point)
        self.assertTrue(result)

        point = self.space.random_point(10)
        result = self.space.belongs(point)
        self.assertTrue(gs.all(result))

    def test_permute(self):
        """Test of permuting method."""
        graph_to_permute = gs.array([[0.0, 1.0], [2.0, 3.0]])
        permutation = [1, 0]
        expected = gs.array([[3.0, 2.0], [1.0, 0.0]])
        permuted = self.space.permute(graph_to_permute, permutation)
        self.assertTrue(gs.all(expected == permuted))

    def test_matchers(self):
        """Test of random_point and belongs methods."""
        set1 = gs.array([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 0.0], [0.0, 1.0]]])
        set2 = gs.array([[[3.0, 2.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]])
        d1 = self.metric.dist(set1, set2, matcher="FAQ")
        d2 = self.metric.dist(set1, set2, matcher="ID")
        self.assertTrue(d1[0] < d2[0])
        self.assertTrue(d1[1] == d2[1])
