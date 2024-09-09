import random

import pytest

from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.open_hemisphere import (
    OpenHemisphere,
    OpenHemispheresProduct,
    OpenHemisphereToHyperboloidDiffeo,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.product_manifold import ProductManifoldTestCase
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.base import OpenSetTestData
from .data.diffeo import DiffeoTestData
from .data.open_hemisphere import OpenHemispherePullbackMetricTestData
from .data.product_manifold import (
    ProductManifoldTestData,
    ProductRiemannianMetricTestData,
)


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
    space = request.cls.space = OpenHemisphere(dim)
    request.cls.data_generator = RandomDataGenerator(space, amplitude=5.0)


@pytest.mark.usefixtures("equipped_open_hemispheres")
class TestOpenHemispherePullbackMetric(
    PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = OpenHemispherePullbackMetricTestData()


class TestOpenHemispheresProduct(
    ProductManifoldTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(3, 5)
    space = OpenHemispheresProduct(_n, equip=False)
    testing_data = ProductManifoldTestData()


class TestOpenHemispheresProductMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(3, 5)
    space = OpenHemispheresProduct(_n)
    data_generator = RandomDataGenerator(space, amplitude=5.0)
    testing_data = ProductRiemannianMetricTestData()
