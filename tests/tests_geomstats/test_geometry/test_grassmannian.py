import random

import pytest

from geomstats.geometry.grassmannian import Grassmannian, GrassmannianCanonicalMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.grassmannian import (
    GrassmannianCompactnessTestCase,
    GrassmannianConnectednessTestCase,
)
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.base import LevelSetTestData
from .data.grassmannian import (
    Grassmannian32TestData,
    GrassmannianCanonicalMetric32TestData,
    GrassmannianCanonicalMetricTestData,
)


def _get_random_params():
    while True:
        a = random.randint(2, 6)
        b = random.randint(2, 6)

        if a != b:
            break

    if a > b:
        n, p = a, b
    else:
        n, p = b, a

    return n, p


@pytest.fixture(
    scope="class",
    params=[(3, 2), _get_random_params()],
)
def spaces(request):
    n, p = request.param
    request.cls.space = Grassmannian(n=n, p=p, equip=False)


@pytest.mark.usefixtures("spaces")
class TestGrassmannian(LevelSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = LevelSetTestData()


@pytest.mark.smoke
class TestGrassmannian32(LevelSetTestCase, metaclass=DataBasedParametrizer):
    space = Grassmannian(3, 2, equip=False)
    testing_data = Grassmannian32TestData()


@pytest.fixture(
    scope="class",
    params=[
        (3, 2),
        _get_random_params(),
    ],
)
def equipped_spaces(request):
    n, p = request.param
    space = Grassmannian(n=n, p=p, equip=False)
    request.cls.space = space
    space.equip_with_metric(GrassmannianCanonicalMetric)


@pytest.mark.usefixtures("equipped_spaces")
class TestGrassmannianCanonicalMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = GrassmannianCanonicalMetricTestData()


@pytest.mark.smoke
class TestGrassmannianCanonicalMetric32(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = Grassmannian(3, 2)
    testing_data = GrassmannianCanonicalMetric32TestData()


@pytest.mark.usefixtures("spaces")
class TestGrassmannianConnectedness(GrassmannianConnectednessTestCase):
    pass


@pytest.mark.usefixtures("spaces")
class TestGrassmannianCompactness(GrassmannianCompactnessTestCase):
    pass
