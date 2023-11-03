import random

import pytest

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.kmeans import RiemannianKMeans
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    ClusterMixinsTestCase,
)
from geomstats.test_cases.learning.kmeans import (
    AgainstFrechetMeanTestCase,
    ClusterInitializationTestCase,
)

from .data.kmeans import (
    AgainstFrechetMeanTestData,
    ClusterInitializationTestData,
    RiemannianKMeansTestData,
)


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
def init_estimators(request):
    space, n_clusters, init = request.param
    request.cls.estimator = RiemannianKMeans(space, n_clusters=n_clusters, init=init)


@pytest.mark.usefixtures("init_estimators")
class TestClusterInitialization(
    ClusterInitializationTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ClusterInitializationTestData()


@pytest.fixture(
    scope="class",
    params=[
        (Hypersphere(dim=random.randint(3, 4)), random.randint(2, 4)),
        (SPDMatrices(n=random.randint(2, 4)), random.randint(2, 4)),
    ],
)
def estimators(request):
    space, n_clusters = request.param
    request.cls.estimator = RiemannianKMeans(space, n_clusters=n_clusters)


@pytest.mark.usefixtures("estimators")
class TestRiemannianKMeans(
    ClusterMixinsTestCase, BaseEstimatorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = RiemannianKMeansTestData()


@pytest.fixture(
    scope="class",
    params=(
        Hypersphere(dim=random.randint(3, 4)),
        SPDMatrices(n=random.randint(2, 4)),
    ),
)
def estimators_one_cluster(request):
    space = request.param
    request.cls.estimator = RiemannianKMeans(space, n_clusters=1, tol=1e-4)
    request.cls.other_estimator = FrechetMean(space, method="adaptive")


@pytest.mark.usefixtures("estimators_one_cluster")
class TestAgainstFrechetMean(
    AgainstFrechetMeanTestCase, metaclass=DataBasedParametrizer
):
    testing_data = AgainstFrechetMeanTestData()
