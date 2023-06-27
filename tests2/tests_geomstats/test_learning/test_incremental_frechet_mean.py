import random

import pytest

import geomstats.backend as gs
from geomstats.distributions.lognormal import LogNormal
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDLogEuclideanMetric,
    SPDMatrices,
)
from geomstats.learning.incremental_frechet_mean import IncrementalFrechetMean
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.learning.incremental_frechet_mean import (
    IncrementalFrechetMeanTestCase,
)

from .data.incremental_frechet_mean import (
    IncrementalFrechetMeanEuclideanTestData,
    IncrementalFrechetMeanTestData,
)


class LogNormalRandomGenerator(RandomDataGenerator):
    def __init__(self, space):
        super().__init__(space)
        self._instantiate_log_normal_sampler()

    def _instantiate_log_normal_sampler(self):
        n = self.space.n
        mean = 2 * gs.eye(n)
        spd_cov_n = (n * (n + 1)) // 2

        cov = gs.eye(spd_cov_n)

        self._log_normal_sampler = LogNormal(self.space, mean, cov)

    def random_point(self, n_points=1):
        return self._log_normal_sampler.sample(n_points)


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

    request.cls.data_generator = LogNormalRandomGenerator(space)


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
