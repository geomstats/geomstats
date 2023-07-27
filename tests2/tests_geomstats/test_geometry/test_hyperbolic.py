import random

import pytest

from geomstats.geometry._hyperbolic import _Hyperbolic
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.hyperbolic import (
    BallToHalfSpace,
    HyperbolicCoordsTransformTestCase,
    HyperbolicTransformer,
)
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
)

from .data.hyperbolic import (
    HyperbolicCmpWithTransformTestData,
    HyperbolicCoordsTransform2TestData,
    HyperbolicCoordsTransformTestData,
)
from .data.riemannian_metric import RiemannianMetricCmpWithPointTransformTestData


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def dims(request):
    request.cls.dim = request.param
    request.cls.space = _Hyperbolic


@pytest.mark.usefixtures("dims")
class TestHyperbolicCoordsTransform(
    HyperbolicCoordsTransformTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HyperbolicCoordsTransformTestData()


@pytest.mark.smoke
class TestHyperbolicCoordsTransform2(
    HyperbolicCoordsTransformTestCase, metaclass=DataBasedParametrizer
):
    space = _Hyperbolic
    testing_data = HyperbolicCoordsTransform2TestData()


@pytest.fixture(
    scope="class",
    params=[
        (random.randint(3, 5), Hyperboloid, PoincareBall),
        (random.randint(3, 5), Hyperboloid, PoincareHalfSpace),
    ],
)
def point_cmp_spaces(request):
    dim, Space, OtherSpace = request.param
    space = request.cls.space = Space(dim)
    other_space = request.cls.other_space = OtherSpace(dim)

    request.cls.point_transformer = HyperbolicTransformer(space, other_space)


@pytest.mark.usefixtures("point_cmp_spaces")
class TestHyperbolicCmpWithPointTransform(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    testing_data = RiemannianMetricCmpWithPointTransformTestData()


@pytest.fixture(
    scope="class",
    params=[
        (random.randint(3, 5), PoincareBall, PoincareHalfSpace),
    ],
)
def cmp_spaces(request):
    dim, Space, OtherSpace = request.param
    space = request.cls.space = Space(dim)
    other_space = request.cls.other_space = OtherSpace(dim)

    request.cls.point_transformer = BallToHalfSpace(space, other_space)


@pytest.mark.usefixtures("cmp_spaces")
class TestHyperbolicCmpWithTransform(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HyperbolicCmpWithTransformTestData()
