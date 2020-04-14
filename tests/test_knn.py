"""Unit tests for the KNN classifier."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.knn import KNearestNeighborsClassifier

TOLERANCE = 1e-4


class TestKNearestNeighborsClassifier(geomstats.tests.TestCase):
    """Class defining the KNN tests."""

    def setUp(self):
        """Define the parameters to test."""
        gs.random.seed(1234)
        self.n_neighbors = 3
        self.dimension = 2
        self.space = Euclidean(dim=self.dimension)
        self.distance = self.space.metric.dist

    @geomstats.tests.np_only
    def test_predict(self):
        """Test the 'predict' class method."""
        training_dataset = gs.array([[0], [1], [2], [3]])
        labels = [0, 0, 1, 1]

        neigh = KNearestNeighborsClassifier(n_neighbors=self.n_neighbors,
                                            distance=self.distance)
        neigh.fit(training_dataset, labels)
        result = neigh.predict([[1.1]])
        expected = gs.array([0])
        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_predict_proba(self):
        """Test the 'predict_proba' class method."""
        training_dataset = gs.array([[0], [1], [2], [3]])
        labels = [0, 0, 1, 1]
        neigh = KNearestNeighborsClassifier(n_neighbors=self.n_neighbors,
                                            distance=self.distance)
        neigh.fit(training_dataset, labels)
        result = neigh.predict_proba([[0.9]])
        expected = gs.array([[2 / 3, 1 / 3]])
        self.assertAllClose(expected, result, atol=TOLERANCE)
