import random

import pytest

from geomstats.geometry.hyperbolic import Hyperbolic, HyperbolicDiffeo
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
)

from .data.diffeo import DiffeoTestData
from .data.hyperbolic import ExtrinsicToBall3TestData, HalfSpaceToBall2TestData
from .data.riemannian_metric import RiemannianMetricCmpWithPointTransformTestData


@pytest.fixture(
    scope="class",
    params=[
        (random.randint(2, 5), "ball", "half-space"),
        (random.randint(2, 5), "ball", "extrinsic"),
        (random.randint(2, 5), "extrinsic", "half-space"),
    ],
)
def diffeos(request):
    dim, from_, to = request.param

    request.cls.space = Hyperbolic(dim, from_, equip=False)
    request.cls.image_space = Hyperbolic(dim, to, equip=False)

    request.cls.diffeo = HyperbolicDiffeo(from_, to)


@pytest.mark.usefixtures("diffeos")
class TestHyperbolicDiffeo(DiffeoTestCase, metaclass=DataBasedParametrizer):
    testing_data = DiffeoTestData()


@pytest.mark.smoke
class TestHalfSpaceToBall2(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _from = "half-space"
    _to = "ball"

    space = Hyperbolic(2, _from, equip=False)
    image_space = Hyperbolic(2, _to, equip=False)

    diffeo = HyperbolicDiffeo(_from, _to)

    testing_data = HalfSpaceToBall2TestData()


@pytest.mark.smoke
class TestExtrinsicToBall3(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _from = "extrinsic"
    _to = "ball"

    space = Hyperbolic(3, _from, equip=False)
    image_space = Hyperbolic(3, _to, equip=False)

    diffeo = HyperbolicDiffeo(_from, _to)

    testing_data = ExtrinsicToBall3TestData()


@pytest.fixture(
    scope="class",
    params=[
        (random.randint(3, 5), "extrinsic", "ball"),
        (random.randint(3, 5), "extrinsic", "half-space"),
        (random.randint(3, 5), "ball", "half-space"),
    ],
)
def cmp_spaces(request):
    dim, from_, to = request.param

    request.cls.space = Hyperbolic(dim, from_)
    request.cls.other_space = Hyperbolic(dim, to)

    request.cls.diffeo = HyperbolicDiffeo(from_, to)


@pytest.mark.usefixtures("cmp_spaces")
class TestHyperbolicCmpWithOther(
    RiemannianMetricComparisonTestCase, metaclass=DataBasedParametrizer
):
    testing_data = RiemannianMetricCmpWithPointTransformTestData()
