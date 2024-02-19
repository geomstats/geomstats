import random

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.flat_riemannian_metric import FlatRiemannianMetric
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.flat_riemannian_metric import (
    FlatRiemannianMetricTestCase,
)

from .data.flat_riemannian_metric import FlatRiemannianMetricTestData


class TestFlatRiemannianMetric(
    FlatRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    _dim = random.randint(2, 5)
    space = Euclidean(_dim, equip=False).equip_with_metric(
        FlatRiemannianMetric,
        metric_matrix=SPDMatrices(_dim, equip=False).random_point(),
    )
    testing_data = FlatRiemannianMetricTestData()
