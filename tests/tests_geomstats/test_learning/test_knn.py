import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.stratified.bhv_space import TreeSpace
from geomstats.learning.knn import KNearestNeighborsClassifier
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.knn import (
    KNearestNeighborsClassifierTestCase,
    NeighborClassifierTestCase,
)

from .data.knn import (
    KNearestNeighborsClassifierEuclideanTestData,
    NeighborClassifierTestData,
)


@pytest.fixture(
    scope="class",
    params=[Euclidean(dim=3), Matrices(3, 3)]
    + ([TreeSpace(n_labels=5)] if not gs.__name__.endswith("pytorch") else []),
)
def estimators(request):
    space = request.param
    request.cls.estimator = KNearestNeighborsClassifier(
        space,
        n_neighbors=1,
    )


@pytest.mark.usefixtures("estimators")
class TestNeighborClassifier(
    NeighborClassifierTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = NeighborClassifierTestData()


@pytest.mark.smoke
class TestKNearestNeighborsClassifier(
    KNearestNeighborsClassifierTestCase,
    metaclass=DataBasedParametrizer,
):
    space = Euclidean(dim=1)
    estimator = KNearestNeighborsClassifier(space, n_neighbors=3)

    testing_data = KNearestNeighborsClassifierEuclideanTestData()
