import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDLogEuclideanMetric,
    SPDMatrices,
)
from geomstats.learning.incremental_frechet_mean import IncrementalFrechetMean
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.incremental_frechet_mean import (
    IncrementalFrechetMeanTestCase,
)

from .data.incremental_frechet_mean import (
    IncrementalFrechetMeanEuclideanTestData,
    IncrementalFrechetMeanTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        SPDMatrices(random.randint(3, 5), equip=False).equip_with_metric(
            SPDAffineMetric
        ),
        SPDMatrices(random.randint(3, 5), equip=False).equip_with_metric(
            SPDLogEuclideanMetric
        ),
    ],
)
def estimators(request):
    space = request.param
    request.cls.estimator = IncrementalFrechetMean(space)


@pytest.mark.usefixtures("estimators")
class TestIncrementalFrechetMean(
    IncrementalFrechetMeanTestCase, metaclass=DataBasedParametrizer
):
    testing_data = IncrementalFrechetMeanTestData()


class TestIncrementalFrechetMeanEuclidean(
    IncrementalFrechetMeanTestCase, metaclass=DataBasedParametrizer
):
    estimator = IncrementalFrechetMean(Euclidean(dim=random.randint(2, 5)))
    testing_data = IncrementalFrechetMeanEuclideanTestData()

    def test_fit_eye(self, atol):
        n = self.estimator.space.dim

        X = gs.eye(n)
        expected = gs.ones(n) / n

        self.test_fit(X, expected, atol)
