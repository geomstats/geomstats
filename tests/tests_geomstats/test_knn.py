"""Unit tests for the KNN classifier."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.knn import KNearestNeighborsClassifier


@tests.conftest.np_and_autograd_only
class TestKNearestNeighborsClassifier(tests.conftest.TestCase):
    """Class defining the KNN tests."""

    def setup_method(self):
        """Define the parameters to test."""
        gs.random.seed(1234)
        self.n_neighbors = 3

    def test_predict(self):
        """Test the 'predict' class method."""
        space = Euclidean(dim=1)
        training_dataset = gs.array([[0.0], [1.0], [2.0], [3.0]])
        labels = [0, 0, 1, 1]

        neigh = KNearestNeighborsClassifier(space, n_neighbors=self.n_neighbors)
        neigh.fit(training_dataset, labels)
        result = neigh.predict([[1.1]])
        expected = gs.array([0])
        self.assertAllClose(expected, result)

    def test_predict_proba(self):
        """Test the 'predict_proba' class method."""
        space = Euclidean(dim=1)
        training_dataset = gs.array([[0.0], [1.0], [2.0], [3.0]])
        labels = [0, 0, 1, 1]
        neigh = KNearestNeighborsClassifier(
            space,
            n_neighbors=self.n_neighbors,
        )
        neigh.fit(training_dataset, labels)
        result = neigh.predict_proba([[0.9]])
        expected = gs.array([[2 / 3, 1 / 3]])
        self.assertAllClose(expected, result, atol=gs.atol)
