import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.hermitian import Hermitian
from geomstats.test_cases.geometry.hermitian import HermitianTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.hermitian import Hermitian2MetricTestData, HermitianTestData
from .data.riemannian_metric import RiemannianMetricTestData


class TestHermitian(HermitianTestCase, metaclass=GeometricDataBasedParametrizer):
    space = Hermitian(dim=random.randint(2, 5), equip=False)
    testing_data = HermitianTestData(space)


@pytest.mark.smoke
class TestHermitian2Metric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = Hermitian(dim=2, equip=True)
    testing_data = Hermitian2MetricTestData()


class TestHermitianMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = Hermitian(dim=random.randint(2, 5))
    testing_data = RiemannianMetricTestData(space)
