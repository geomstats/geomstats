import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

import geomstats.backend as gs
from geomstats.geometry.functions import HilbertSphere
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
    RiemannianMetricTestCase,
)
from geomstats.variants.geometry.functions import TrapezoidHilbertSphereMetric

from .data.functions import HilbertSphereSmokeTestData
from .data.manifold import ManifoldTestData
from .data.riemannian_metric import (
    RiemannianMetricComparisonTestData,
    RiemannianMetricTestData,
)


@pytest.mark.smoke
class TestHilbertSphereSmoke(ManifoldTestCase, metaclass=DataBasedParametrizer):
    space = HilbertSphere(gs.linspace(0, 1, num=50))
    testing_data = HilbertSphereSmokeTestData()


class TestHilbertSphere(ManifoldTestCase, metaclass=GeometricDataBasedParametrizer):
    space = HilbertSphere(gs.linspace(0, 1, num=random.randint(5, 10)))
    testing_data = ManifoldTestData(space)


class TestHilbertSphereMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = HilbertSphere(gs.linspace(0, 1, num=random.randint(5, 10)))
    testing_data = RiemannianMetricTestData(space)


@pytest.mark.redundant
class TestTrapezoidHilbertSphereMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = HilbertSphere(
        gs.linspace(0, 1, num=random.randint(5, 10)), equip=False
    ).equip_with_metric(TrapezoidHilbertSphereMetric)
    testing_data = RiemannianMetricTestData(space)


@pytest.mark.redundant
class TestTrapezoidHilbertSphereMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    _domain = gs.linspace(-1, 2, num=random.randint(5, 10))

    space = HilbertSphere(_domain)
    other_space = HilbertSphere(_domain, equip=False).equip_with_metric(
        TrapezoidHilbertSphereMetric
    )
    testing_data = RiemannianMetricComparisonTestData(space)
