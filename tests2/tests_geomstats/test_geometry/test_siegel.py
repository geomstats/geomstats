import random

import pytest

from geomstats.geometry.siegel import Siegel, SiegelMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import ComplexOpenSetTestCase
from geomstats.test_cases.geometry.siegel import SiegelMetricTestCase

from .data.base import ComplexOpenSetTestData
from .data.siegel import (
    Siegel2MetricTestData,
    Siegel2TestData,
    Siegel3MetricTestData,
    SiegelMetricTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = Siegel(request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestSiegel(ComplexOpenSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = ComplexOpenSetTestData()


@pytest.mark.smoke
class TestSiegel2(ComplexOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = Siegel(2, equip=False)
    testing_data = Siegel2TestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_spaces(request):
    space = request.cls.space = Siegel(request.param, equip=False)
    space.equip_with_metric(SiegelMetric)


@pytest.mark.usefixtures("equipped_spaces")
class TestSiegelMetric(SiegelMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = SiegelMetricTestData()


@pytest.mark.smoke
class TestSiegel2Metric(SiegelMetricTestCase, metaclass=DataBasedParametrizer):
    space = Siegel(2, equip=True)
    testing_data = Siegel2MetricTestData()


@pytest.mark.smoke
class TestSiegel3Metric(SiegelMetricTestCase, metaclass=DataBasedParametrizer):
    space = Siegel(3, equip=True)
    testing_data = Siegel3MetricTestData()
