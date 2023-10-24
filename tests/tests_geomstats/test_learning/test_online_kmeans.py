import random

import pytest

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.online_kmeans import OnlineKMeans
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    ClusterMixinsTestCase,
)

from .data.online_kmeans import OnlineKMeansTestData


@pytest.fixture(
    scope="class",
    params=[
        (Hypersphere(dim=random.randint(3, 4)), random.randint(2, 4)),
        (SPDMatrices(n=random.randint(2, 4)), random.randint(2, 4)),
    ],
)
def estimators(request):
    space, n_clusters = request.param
    request.cls.estimator = OnlineKMeans(space, n_clusters=n_clusters)


@pytest.mark.usefixtures("estimators")
class TestRiemannianKMeans(
    ClusterMixinsTestCase, BaseEstimatorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = OnlineKMeansTestData()
