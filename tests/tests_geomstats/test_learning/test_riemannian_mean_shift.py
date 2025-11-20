import random

import pytest

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.riemannian_mean_shift import RiemannianMeanShift
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    ClusterMixinsTestCase,
)

from .data.riemannian_mean_shift import RiemannianMeanShiftTestData


@pytest.fixture(
    scope="class",
    params=[
        (Hypersphere(dim=2), 0.6, random.randint(2, 4)),
    ],
)
def estimators(request):
    space, bandwidth, n_clusters = request.param
    request.cls.estimator = RiemannianMeanShift(
        space,
        bandwidth,
        n_clusters=n_clusters,
    )


@pytest.mark.usefixtures("estimators")
class TestRiemannianMeanShift(
    ClusterMixinsTestCase, BaseEstimatorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = RiemannianMeanShiftTestData()
