import pytest

from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.knn import KNearestNeighborsClassifier
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.knn import KNearestNeighborsClassifierTestCase

from .data.knn import KNearestNeighborsClassifierEuclideanTestData


@pytest.mark.smoke
class TestKNearestNeighborsClassifier(
    KNearestNeighborsClassifierTestCase,
    metaclass=DataBasedParametrizer,
):
    space = Euclidean(dim=1)
    estimator = KNearestNeighborsClassifier(space, n_neighbors=3)

    testing_data = KNearestNeighborsClassifierEuclideanTestData()
