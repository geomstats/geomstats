import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.functions import HilbertSphere
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.mixins import ProjectionTestCaseMixins
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.functions import (
    HilbertSphereMetricTestData,
    HilbertSphereSmokeTestData,
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
class TestHilbertSphere(
    ProjectionTestCaseMixins, ManifoldTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HilbertSphereTestData()


@pytest.mark.smoke
class TestHilbertSphereSmoke(ManifoldTestCase, metaclass=DataBasedParametrizer):
    space = HilbertSphere(gs.linspace(0, 1, num=50))
    testing_data = HilbertSphereSmokeTestData()


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
    request.cls.space = HilbertSphere(domain)


@pytest.mark.usefixtures("equipped_spaces")
class TestHilbertSphereMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HilbertSphereMetricTestData()
