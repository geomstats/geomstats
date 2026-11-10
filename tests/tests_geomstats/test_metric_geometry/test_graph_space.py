"""Unit tests for the graphspace quotient space."""

import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.metric_geometry.graph_space import (
    ExhaustiveAligner,
    FAQAligner,
    GraphSpace,
    PointToGeodesicAligner,
    _GeodesicToPointAligner,
)
from geomstats.test_cases.geometry.fiber_bundle import AlignComparisonTestCase
from geomstats.test_cases.geometry.matrices import MatricesTestCase
from geomstats.test_cases.metric_geometry.graph_space import (
    PointToGeodesicAlignerTestCase,
)
from geomstats.test_cases.metric_geometry.quotient import (
    AlignerTestCase,
    QuotientMetricWithArrayTestCase,
)

from ..test_geometry.data.fiber_bundle import AlignComparisonTestData
from ..test_geometry.data.matrices import MatricesTestData
from .data.graph_space import PointToGeodesicAlignerTestData
from .data.quotient import AlignerTestData, QuotientMetricWithArrayTestData


def _graph_space_with_quotient():
    total_space = GraphSpace(random.randint(2, 4))
    total_space.equip_with_group_action()
    total_space.equip_with_quotient()
    return total_space


class TestGraphSpace(MatricesTestCase, metaclass=GeometricDataBasedParametrizer):
    space = GraphSpace(random.randint(2, 5), equip=False)
    testing_data = MatricesTestData(space)


@pytest.fixture(scope="class", params=["exhaustive", "FAQ"])
def aligner_algorithms(request):
    total_space = request.cls.total_space = _graph_space_with_quotient()
    total_space.aligner.set_alignment_algorithm(request.param)

    request.cls.testing_data.space = total_space


@pytest.mark.usefixtures("aligner_algorithms")
class TestGraphSpaceAligner(AlignerTestCase, metaclass=GeometricDataBasedParametrizer):
    testing_data = AlignerTestData()


@pytest.mark.xfail
class TestGraphAlignerAlgorithmCmp(
    AlignComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    total_space = _graph_space_with_quotient()

    align = ExhaustiveAligner(total_space).align
    other_align = FAQAligner(total_space).align

    testing_data = AlignComparisonTestData(total_space, point_counts=[1, 2])


class TestPointToGeodesicAligner(
    PointToGeodesicAlignerTestCase, metaclass=GeometricDataBasedParametrizer
):
    total_space = _graph_space_with_quotient()
    total_space.aligner.set_point_to_geodesic_aligner(
        PointToGeodesicAligner(total_space, s_min=0.0, s_max=1.0, n_grid=100)
    )

    testing_data = PointToGeodesicAlignerTestData(
        total_space,
        method_defaults={"dist_along_geodesic_is_zero": {"atol": 1e-2}},
    )


# NB: `DataBasedParametrizer` skips vectorization tests, as the aligner only
# handles one geodesic at a time
class TestGeodesicToPointAligner(
    PointToGeodesicAlignerTestCase, metaclass=DataBasedParametrizer
):
    total_space = _graph_space_with_quotient()
    total_space.aligner.set_point_to_geodesic_aligner(
        _GeodesicToPointAligner(total_space)
    )

    testing_data = PointToGeodesicAlignerTestData(
        total_space,
        point_counts=[1],
        method_defaults={"dist_along_geodesic_is_zero": {"atol": 1e-2}},
    )


class TestGraphSpaceQuotientMetric(
    QuotientMetricWithArrayTestCase, metaclass=DataBasedParametrizer
):
    total_space = _graph_space_with_quotient()
    space = total_space.quotient
    testing_data = QuotientMetricWithArrayTestData(space)
