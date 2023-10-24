import pytest

from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDEuclideanMetric,
    SPDLogEuclideanMetric,
    SPDMatrices,
)
from geomstats.learning.mdm import RiemannianMinimumDistanceToMean
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.mdm import RiemannianMinimumDistanceToMeanTestCase

from .data.mdm import (
    RiemannianMinimumDistanceToMeanSPDEuclideanTestData,
    RiemannianMinimumDistanceToMeanSPDTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        SPDAffineMetric,
        SPDLogEuclideanMetric,
    ],
)
def estimators(request):
    Metric = request.param
    space = SPDMatrices(n=2, equip=False).equip_with_metric(Metric)
    request.cls.estimator = RiemannianMinimumDistanceToMean(space)


@pytest.mark.smoke
@pytest.mark.usefixtures("estimators")
class TestRiemannianMinimumDistanceToMeanSPD(
    RiemannianMinimumDistanceToMeanTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = RiemannianMinimumDistanceToMeanSPDTestData()


@pytest.mark.smoke
class TestRiemannianMinimumDistanceToMeanSPDEuclidean(
    RiemannianMinimumDistanceToMeanTestCase,
    metaclass=DataBasedParametrizer,
):
    space = SPDMatrices(n=2, equip=False).equip_with_metric(
        SPDEuclideanMetric,
    )
    estimator = RiemannianMinimumDistanceToMean(space)
    testing_data = RiemannianMinimumDistanceToMeanSPDEuclideanTestData()
