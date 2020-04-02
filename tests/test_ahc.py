"""Unit tests for the KNN classifier."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.ahc import AgglomerativeHierarchicalClustering


class TestAgglomerativeHierarchicalClustering(geomstats.tests.TestCase):
    """Class defining the Poincare polydisk tests."""

    def setUp(self):
        """Define the parameters to test."""
        gs.random.seed(1234)
        self.n_clusters = 2
        self.dimension = 2
        self.space = Euclidean(dimension=self.dimension)
        self.distance = self.space.metric.dist

    @geomstats.tests.np_only
    def test_fit(self):
        """Test the 'fit' class method."""
        dataset = gs.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        clustering = AgglomerativeHierarchicalClustering(
            n_clusters=self.n_clusters,
            distance=self.distance)
        clustering.fit(dataset)
        clustering_labels = clustering.labels_
        result = ((clustering_labels == gs.array([1, 1, 1, 0, 0, 0])).all() or
                  (clustering_labels == gs.array([0, 0, 0, 1, 1, 1])).all())
        expected = True
        self.assertAllClose(expected, result)
