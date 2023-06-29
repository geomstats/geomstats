import random

import pytest

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDLogEuclideanMetric,
    SPDMatrices,
)
from geomstats.learning.geometric_median import GeometricMedian
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import TestCase
from geomstats.test_cases.learning.geometric_median import GeometricMedianTestCase

from .data.geometric_median import GeometricMedianFitTestData, GeometricMedianTestData


@pytest.fixture(
    scope="class",
    params=[
        Euclidean(random.randint(3, 5)),
        Hyperboloid(random.randint(3, 5)),
        Hypersphere(random.randint(3, 5)),
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
    request.cls.estimator = GeometricMedian(space)


@pytest.mark.usefixtures("estimators")
class TestGeometricMedian(GeometricMedianTestCase, metaclass=DataBasedParametrizer):
    testing_data = GeometricMedianTestData()


@pytest.mark.smoke
class TestGeometricMedianFit(TestCase, metaclass=DataBasedParametrizer):
    testing_data = GeometricMedianFitTestData()

    def test_fit(self, estimator, X, expected, atol):
        estimate = estimator.fit(X).estimate_
        self.assertAllClose(estimate, expected, atol=atol)
