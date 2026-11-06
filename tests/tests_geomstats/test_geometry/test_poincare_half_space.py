import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.poincare_half_space import (
    PoincareHalfSpace,
    PoincareHalfSpaceMetric,
)
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.poincare_half_space import (
    PoincareHalfSpaceMetricTestCase,
)

from .data.poincare_half_space import (
    PoincareHalfSpace2TestData,
    PoincareHalfSpaceMetric2TestData,
    PoincareHalfSpaceTestData,
)
from .data.riemannian_metric import RiemannianMetricTestData


@pytest.mark.smoke
class TestPoincareHalfSpace2(
    VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = PoincareHalfSpace(dim=2, equip=False)
    testing_data = PoincareHalfSpace2TestData()


class TestPoincareHalfSpace(
    VectorSpaceOpenSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PoincareHalfSpace(dim=random.randint(2, 5), equip=False)
    testing_data = PoincareHalfSpaceTestData(space)


@pytest.mark.smoke
class TestPoincareHalfSpaceMetric2(
    PoincareHalfSpaceMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PoincareHalfSpace(dim=2, equip=False)
    space.equip_with_metric(PoincareHalfSpaceMetric)
    testing_data = PoincareHalfSpaceMetric2TestData()


class TestPoincareHalfSpaceMetric(
    PoincareHalfSpaceMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PoincareHalfSpace(dim=random.randint(2, 5))
    testing_data = RiemannianMetricTestData(space)
