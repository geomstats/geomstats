import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.siegel import Siegel, SiegelMetric
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.siegel import SiegelMetricTestCase

from .data.base import VectorSpaceOpenSetTestData
from .data.siegel import (
    Siegel2MetricTestData,
    Siegel2TestData,
    Siegel3MetricTestData,
    SiegelMetricTestData,
)


@pytest.fixture(scope="class", params=[False, True])
def spaces(request):
    space = request.cls.space = Siegel(
        random.randint(2, 5), symmetric=request.param, equip=request.cls._equip
    )

    request.cls.testing_data.space = space


@pytest.mark.usefixtures("spaces")
class TestSiegel(VectorSpaceOpenSetTestCase, metaclass=GeometricDataBasedParametrizer):
    _equip = False
    testing_data = VectorSpaceOpenSetTestData()


@pytest.mark.smoke
class TestSiegel2(VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = Siegel(2, equip=False)
    testing_data = Siegel2TestData()


@pytest.mark.usefixtures("spaces")
class TestSiegelMetric(SiegelMetricTestCase, metaclass=GeometricDataBasedParametrizer):
    _equip = True
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
