import random

from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.geometry.scalar_product_metric import (
    MetricScalingTestCase,
    ScalarProductMetricComparisonTestCase,
    ScaledMethodRegistrationTestCase,
)

from .data.riemannian_metric import RiemannianMetricTestData
from .data.scalar_product_metric import (
    MetricScalingTestData,
    ScalarProductMetricComparisonTestData,
    ScaledMethodRegistrationTestData,
)


class TestScalarProductMetricCmp(
    ScalarProductMetricComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    _dim = random.randint(2, 5)
    scale = random.uniform(0.5, 3.0)
    space = Euclidean(dim=_dim)
    other_space = Euclidean(dim=_dim).equip_with_metric(
        ScalarProductMetric, scale=scale
    )
    testing_data = ScalarProductMetricComparisonTestData(space)


class TestScaledEuclideanMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = Euclidean(dim=random.randint(2, 5)).equip_with_metric(
        ScalarProductMetric, scale=random.uniform(0.5, 3.0)
    )
    testing_data = RiemannianMetricTestData(space)


class TestMetricScaling(MetricScalingTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(dim=3)
    testing_data = MetricScalingTestData()


class TestScaledMethodRegistration(
    ScaledMethodRegistrationTestCase, metaclass=DataBasedParametrizer
):
    Metric = EuclideanMetric
    space = Euclidean(dim=3)
    testing_data = ScaledMethodRegistrationTestData()
