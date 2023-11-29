import random

import pytest

from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.open_hemisphere import (
    OpenHemisphere,
    OpenHemisphereToHyperboloidDiffeo,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase

from .data.base import OpenSetTestData
from .data.diffeo import DiffeoTestData
from .data.open_hemisphere import OpenHemispherePullbackMetricTestData


@pytest.fixture(
    scope="class",
    params=[
        1,
        random.randint(2, 4),
    ],
)
def diffeos(request):
    dim = request.param
    request.cls.space = OpenHemisphere(dim, equip=False)
    request.cls.image_space = Hyperboloid(dim, equip=False)


@pytest.mark.usefixtures("diffeos")
class TestOpenHemisphereToHyperboloidDiffeo(
    DiffeoTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 5)
    diffeo = OpenHemisphereToHyperboloidDiffeo()
    testing_data = DiffeoTestData()


@pytest.fixture(
    scope="class",
    params=[
        1,
        random.randint(2, 4),
    ],
)
def open_hemispheres(request):
    dim = request.param
    request.cls.space = OpenHemisphere(dim, equip=False)


@pytest.mark.usefixtures("open_hemispheres")
class TestOpenHemisphere(OpenSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = OpenSetTestData()


@pytest.fixture(
    scope="class",
    params=[
        1,
        random.randint(2, 4),
    ],
)
def equipped_open_hemispheres(request):
    dim = request.param
    request.cls.space = OpenHemisphere(dim)


@pytest.mark.usefixtures("equipped_open_hemispheres")
class TestOpenHemispherePullbackMetric(
    PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = OpenHemispherePullbackMetricTestData()
