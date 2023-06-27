import random

import pytest

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.kmeans import RiemannianKMeans
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.kmeans import RiemannianKMeansTestCase

from .data.kmeans import RiemannianKMeansOneClusterTestData, RiemannianKMeansTestData


@pytest.fixture(
    scope="class",
    params=[
        Hypersphere(dim=random.randint(3, 4)),
        SPDMatrices(n=random.randint(2, 4)),
    ],
)
def one_cluster_estimators(request):
    space = request.param
    request.cls.estimator = RiemannianKMeans(space, n_clusters=1)


@pytest.mark.usefixtures("one_cluster_estimators")
class TestRiemannianKMeansOneCluster(
    RiemannianKMeansTestCase, metaclass=DataBasedParametrizer
):
    testing_data = RiemannianKMeansOneClusterTestData()


def _make_centroids(space, X, n_clusters):
    return space.random_von_mises_fisher(kappa=10, n_samples=n_clusters)


def _get_params():
    params = []
    spaces = (
        Hypersphere(dim=random.randint(3, 4)),
        SPDMatrices(n=random.randint(2, 4)),
    )

    init = ["kmeans++", "random"]
    for space in spaces:
        n_clusters = random.randint(2, 4)
        for init_ in init:
            params.append((space, n_clusters, init_))

    params.append(
        (
            spaces[0],
            params[0][1],
            lambda X, n_clusters: _make_centroids(spaces[0], X, n_clusters),
        )
    )

    return params


@pytest.fixture(
    scope="class",
    params=_get_params(),
)
def estimators(request):
    space, n_clusters, init = request.param
    request.cls.estimator = RiemannianKMeans(space, n_clusters=n_clusters, init=init)


@pytest.mark.usefixtures("estimators")
class TestRiemannianKMeans(RiemannianKMeansTestCase, metaclass=DataBasedParametrizer):
    testing_data = RiemannianKMeansTestData()
