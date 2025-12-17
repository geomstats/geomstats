import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDLogEuclideanMetric,
    SPDMatrices,
)
from geomstats.learning.mds import MetricMDS
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.mds import (
    MetricMDSTestCase,
)

from .data.mds import (
    MetricMDSEuclideanTestData,
    MetricMDSTestData,
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
    request.cls.estimator = MetricMDS(space)


@pytest.mark.usefixtures("estimators")
class TestMetricMDS(MetricMDSTestCase, metaclass=DataBasedParametrizer):
    testing_data = MetricMDSTestData()


class TestMetricMDSEuclidean(MetricMDSTestCase, metaclass=DataBasedParametrizer):
    estimator = MetricMDS(
        Euclidean(dim=random.randint(2, 5)), n_components=random.randint(2, 3)
    )
    testing_data = MetricMDSEuclideanTestData()

    def test_fit_eye(self, atol):
        n = self.estimator.space.dim

        X = gs.eye(n)
        expected = gs.ones(n) - gs.eye(n)

        self.test_dissimilarity_matrix(X, expected, atol)
