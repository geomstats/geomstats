import random

import pytest

from geomstats.geometry.discrete_curves import (
    ClosedDiscreteCurves,
    DiscreteCurves,
    SRVShapeBundle,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.test.geometry.discrete_curves import (
    ClosedDiscreteCurvesTestCase,
    DiscreteCurvesTestCase,
    SRVQuotientMetricTestCase,
    SRVShapeBundleTestCase,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.discrete_curves_data import (
    ClosedDiscreteCurvesTestData,
    DiscreteCurvesTestData,
    SRVQuotientMetricTestData,
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
def shape_bundles(request):
    dim, k_sampling_points = request.param

    ambient_manifold = Euclidean(dim=dim)
    # TODO: can also test different metrics
    space = DiscreteCurves(
        ambient_manifold, k_sampling_points=k_sampling_points, equip=True
    )
    request.cls.total_space = request.cls.base = space

    request.cls.bundle = SRVShapeBundle(space)

    request.cls.sphere = Hypersphere(dim=dim - 1)


@pytest.mark.usefixtures("shape_bundles")
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


@pytest.fixture(
    scope="class",
    params=[
        (2, random.randint(5, 10)),
        # (3, random.randint(5, 10)),
    ],
)
def spaces_with_quotient(request):
    dim, k_sampling_points = request.param

    ambient_manifold = Euclidean(dim=dim)
    space = DiscreteCurves(
        ambient_manifold, k_sampling_points=k_sampling_points, equip=True
    )

    space.equip_with_group_action("reparametrizations")
    space.equip_with_quotient_structure()

    request.cls.space = space.quotient

    request.cls.sphere = Hypersphere(dim=dim - 1)


@pytest.mark.usefixtures("spaces_with_quotient")
class TestSRVQuotientMetric(SRVQuotientMetricTestCase, metaclass=DataBasedParametrizer):
    # TODO: failing. need to understand why
    testing_data = SRVQuotientMetricTestData()
