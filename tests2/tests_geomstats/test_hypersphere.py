import random

import pytest

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.test.geometry.hypersphere import (
    HypersphereCoordsTransformTestCase,
    HypersphereExtrinsicTestCase,
    HypersphereIntrinsicTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.hypersphere_data import (
    HypersphereCoordsTransformTestData,
    HypersphereExtrinsicTestData,
    HypersphereIntrinsicTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        1,
        2,
    ],
)
def coords_transform_spaces(request):
    dim = request.param
    request.cls.space_intrinsic = Hypersphere(dim, default_coords_type="intrinsic")
    request.cls.space = request.cls.space_extrinsic = Hypersphere(
        dim, default_coords_type="extrinsic"
    )


@pytest.mark.usefixtures("coords_transform_spaces")
class TestHypersphereCoordsTransform(
    HypersphereCoordsTransformTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HypersphereCoordsTransformTestData()


@pytest.fixture(
    scope="class",
    params=[
        1,
        2,
        random.randint(3, 5),
    ],
)
def extrinsic_spaces(request):
    dim = request.param
    request.cls.space = Hypersphere(dim, default_coords_type="extrinsic")


@pytest.mark.usefixtures("extrinsic_spaces")
class TestHypersphereExtrinsic(
    HypersphereExtrinsicTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HypersphereExtrinsicTestData()


@pytest.fixture(
    scope="class",
    params=[
        1,
        2,
    ],
)
def intrinsic_spaces(request):
    dim = request.param
    request.cls.space = Hypersphere(dim, default_coords_type="intrinsic")


@pytest.mark.usefixtures("intrinsic_spaces")
class TestHypersphereIntrinsic(
    HypersphereIntrinsicTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HypersphereIntrinsicTestData()
