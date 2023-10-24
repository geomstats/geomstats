import pytest

import geomstats.backend as gs
from geomstats.distributions.lognormal import LogNormal
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.spd_matrices import SPDLogEuclideanMetric, SPDMatrices
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.distributions.lognormal import LogNormalTestCase

from .data.lognormal import LogNormalTestData


@pytest.fixture(
    scope="class",
    params=[
        (Euclidean(dim=3), gs.zeros(3), gs.eye(3)),
        (SPDMatrices(n=3), 2 * gs.eye(3), gs.eye((3 * (3 + 1)) // 2)),
        (
            SPDMatrices(n=3, equip=False).equip_with_metric(SPDLogEuclideanMetric),
            2 * gs.eye(3),
            gs.eye((3 * (3 + 1)) // 2),
        ),
    ],
)
def distributions(request):
    space, mean, cov = request.param

    request.cls.distribution = LogNormal(
        space,
        mean,
        cov,
    )
    request.cls.mean_estimator = FrechetMean(space)


@pytest.mark.usefixtures("distributions")
class TestLogNormal(LogNormalTestCase, metaclass=DataBasedParametrizer):
    testing_data = LogNormalTestData()
