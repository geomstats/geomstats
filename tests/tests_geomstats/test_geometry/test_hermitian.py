import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.hermitian import Hermitian
from geomstats.test_cases.geometry.complex_riemannian_metric import (
    ComplexRiemannianMetricTestCase,
)
from geomstats.test_cases.geometry.hermitian import HermitianTestCase

from .data.complex_riemannian_metric import ComplexRiemannianMetricTestData
from .data.hermitian import Hermitian2MetricTestData, HermitianTestData


class TestHermitian(HermitianTestCase, metaclass=GeometricDataBasedParametrizer):
    space = Hermitian(dim=random.randint(2, 5), equip=False)
    testing_data = HermitianTestData(space)


@pytest.mark.smoke
class TestHermitian2Metric(
    ComplexRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = Hermitian(dim=2, equip=True)
    testing_data = Hermitian2MetricTestData()


class TestHermitianMetric(
    ComplexRiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = Hermitian(dim=random.randint(2, 5))
    testing_data = ComplexRiemannianMetricTestData(space)
