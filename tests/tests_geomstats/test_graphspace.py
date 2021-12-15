"""Unit tests for the graphspace quotient space."""

import geomstats.tests
import tests.helper as helper
import geomstats.backend as gs
from geomstats.geometry.graphspace import GraphSpace, GraphSpaceMetric

class TestGraphSpace(geomstats.tests.TestCase):
    """Test of Graph Space quotient metric space."""

    def setup_method(self):
        """Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

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
        expected = True
        self.assertAllTrue(result, expected)

        vec = gs.array([-1.0, -1.0])
        result = self.space.belongs(vec)
        expected = False
        self.assertAllFalse(result, expected)


    def test_random_point_and_belongs(self):
        """Test of random_point and belongs methods."""
        point = self.space.random_point()
        result = self.space.belongs(point)
        expected = True
        self.assertAllClose(result, expected)

    def test_permute(self):
        """Test of permuting method."""
        return 0
    def test_metric_ID(self):
        """Test of random_point and belongs methods."""
        return 0