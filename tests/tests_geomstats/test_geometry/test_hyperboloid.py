import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.hyperboloid import HyperboloidMetricTestCase

from .data.base import LevelSetTestData
from .data.hyperboloid import (
    Hyperboloid2TestData,
    Hyperboloid3TestData,
    HyperboloidMetricTestData,
)


@pytest.mark.smoke
class TestHyperboloid2(LevelSetTestCase, metaclass=DataBasedParametrizer):
    space = Hyperboloid(dim=2)
    testing_data = Hyperboloid2TestData()


@pytest.mark.smoke
class TestHyperboloid3(LevelSetTestCase, metaclass=DataBasedParametrizer):
    space = Hyperboloid(dim=3)
    testing_data = Hyperboloid3TestData()


class TestHyperboloid(LevelSetTestCase, metaclass=GeometricDataBasedParametrizer):
    space = Hyperboloid(dim=random.randint(2, 5), equip=False)
    testing_data = LevelSetTestData(
        space, method_defaults={"projection_belongs": {"atol": 1e-8}}
    )


class TestHyperboloidMetric(
    HyperboloidMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = Hyperboloid(dim=random.randint(2, 5))
    testing_data = HyperboloidMetricTestData(space)
