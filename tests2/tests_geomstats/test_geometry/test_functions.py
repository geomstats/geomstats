import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.functions import HilbertSphere, HilbertSphereMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.functions import (
    HilbertSphereMetricTestCase,
    HilbertSphereTestCase,
)
from tests2.tests_geomstats.test_geometry.data.functions import (
    HilbertSphereMetricTestData,
    HilbertSphereTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        5,
        random.randint(5, 10),
    ],
)
def spaces(request):
    n_samples = request.param
    domain = gs.linspace(0, 1, num=n_samples)
    request.cls.space = HilbertSphere(domain, equip=True)


@pytest.mark.usefixtures("spaces")
class TestHilbertSphere(HilbertSphereTestCase, metaclass=DataBasedParametrizer):
    testing_data = HilbertSphereTestData()


@pytest.fixture(
    scope="class",
    params=[
        5,
        random.randint(5, 10),
    ],
)
def equipped_spaces(request):
    n_samples = request.param
    domain = gs.linspace(0, 1, num=n_samples)
    space = request.cls.space = HilbertSphere(domain, equip=False)
    space.equip_with_metric(HilbertSphereMetric)


@pytest.mark.usefixtures("equipped_spaces")
class TestHilbertSphereMetric(
    HilbertSphereMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HilbertSphereMetricTestData()
