import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.positive_hypersphere import PositiveHypersphere
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.base import OpenSetTestData
from .data.positive_hypersphere import (
    PositiveHypersphere1MetricTestData,
    PositiveHypersphere1TestData,
)
from .data.riemannian_metric import RiemannianMetricTestData


class TestPositiveHypersphere(
    OpenSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveHypersphere(dim=random.randint(2, 5), equip=False)
    testing_data = OpenSetTestData(space)


@pytest.mark.smoke
class TestPositiveHypersphere1(OpenSetTestCase, metaclass=DataBasedParametrizer):
    space = PositiveHypersphere(dim=1, equip=False)
    testing_data = PositiveHypersphere1TestData()


class TestPositiveHypersphereMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveHypersphere(dim=random.randint(2, 5))
    testing_data = RiemannianMetricTestData(
        space, generator_kwargs={"within_injectivity_radius": True}
    )


@pytest.mark.smoke
class TestPositiveHypersphere1Metric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PositiveHypersphere(dim=1)
    testing_data = PositiveHypersphere1MetricTestData()
