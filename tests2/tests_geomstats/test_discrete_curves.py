import random

import pytest

from geomstats.geometry.discrete_curves import (
    ClosedDiscreteCurves,
    DiscreteCurves,
    SRVPreShapeSpace,
    SRVShapeBundle,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.test.geometry.discrete_curves import (
    ClosedDiscreteCurvesTestCase,
    DiscreteCurvesTestCase,
    SRVShapeBundleTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.discrete_curves_data import (
    ClosedDiscreteCurvesTestData,
    DiscreteCurvesTestData,
    SRVShapeBundleTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        (2, random.randint(5, 10)),
        (3, random.randint(5, 10)),
    ],
)
def discrete_curves_spaces(request):
    dim, k_sampling_points = request.param

    ambient_manifold = Euclidean(dim=dim)
    request.cls.space = DiscreteCurves(
        ambient_manifold, k_sampling_points=k_sampling_points
    )


@pytest.mark.usefixtures("discrete_curves_spaces")
class TestDiscreteCurves(DiscreteCurvesTestCase, metaclass=DataBasedParametrizer):
    testing_data = DiscreteCurvesTestData()


@pytest.fixture(
    scope="class",
    params=[
        (2, random.randint(5, 10)),
        (3, random.randint(5, 10)),
    ],
)
def bundles_spaces(request):
    dim, k_sampling_points = request.param

    ambient_manifold = Euclidean(dim=dim)
    base = SRVPreShapeSpace(
        ambient_manifold, k_sampling_points=k_sampling_points, equip=False
    )
    request.cls.base = base
    request.cls.space = base.fiber_bundle
    request.cls.sphere = Hypersphere(dim=dim - 1)


@pytest.mark.usefixtures("bundles_spaces")
class TestSRVShapeBundle(SRVShapeBundleTestCase, metaclass=DataBasedParametrizer):
    testing_data = SRVShapeBundleTestData()


@pytest.fixture(
    scope="class",
    params=[
        (2, random.randint(5, 10)),
        (3, random.randint(5, 10)),
    ],
)
def closed_discrete_curves_spaces(request):
    dim, k_sampling_points = request.param
    ambient_manifold = Euclidean(dim=dim)

    request.cls.space = ClosedDiscreteCurves(
        ambient_manifold, k_sampling_points=k_sampling_points
    )


@pytest.mark.usefixtures("closed_discrete_curves_spaces")
class TestClosedDiscreteCurves(
    ClosedDiscreteCurvesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ClosedDiscreteCurvesTestData()
