import random

import pytest

from geomstats.geometry.siegel import Siegel, SiegelMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import ComplexVectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.siegel import SiegelMetricTestCase

from .data.base import ComplexVectorSpaceOpenSetTestData
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
class TestSiegel(ComplexVectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = ComplexVectorSpaceOpenSetTestData()


@pytest.mark.smoke
class TestSiegel2(ComplexVectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
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
    request.cls.space = Siegel(request.param)


@pytest.mark.usefixtures("equipped_spaces")
class TestSiegelMetric(SiegelMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = SiegelMetricTestData()


@pytest.mark.smoke
class TestSiegel2Metric(SiegelMetricTestCase, metaclass=DataBasedParametrizer):
    space = Siegel(2, equip=False)
    space.equip_with_metric(SiegelMetric)
    testing_data = Siegel2MetricTestData()


@pytest.mark.smoke
class TestSiegel3Metric(SiegelMetricTestCase, metaclass=DataBasedParametrizer):
    space = Siegel(3, equip=False)
    space.equip_with_metric(SiegelMetric)
    testing_data = Siegel3MetricTestData()
