"""Unit tests for the graphspace quotient space."""

import random

import pytest

from geomstats.geometry.stratified.graph_space import (
    ExhaustiveAligner,
    FAQAligner,
    GraphSpace,
    PointToGeodesicAligner,
    _GeodesicToPointAligner,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.matrices import MatricesTestCase
from geomstats.test_cases.geometry.stratified.graph_space import (
    PointToGeodesicAlignerTestCase,
)
from geomstats.test_cases.geometry.stratified.quotient import (
    AlignerAlgorithmCmpTestCase,
    AlignerAlgorithmTestCase,
    QuotientMetricWithArrayTestCase,
)

from ..data.matrices import MatricesTestData
from .data.graph_space import GraphAlignerCmpTestData, PointToGeodesicAlignerTestData
from .data.quotient import AlignerAlgorithmTestData, QuotientMetricWithArrayTestData


@pytest.fixture(
    scope="class",
    params=[
        ExhaustiveAligner,
        FAQAligner,
    ],
)
def aligner_algorithms(request):
    Aligner = request.param

    n = random.randint(2, 4)
    request.cls.total_space = total_space = GraphSpace(n)
    total_space.equip_with_group_action()

    request.cls.aligner = Aligner(total_space)


@pytest.mark.usefixtures("aligner_algorithms")
class TestGraphAlignerAlgorithm(
    AlignerAlgorithmTestCase, metaclass=DataBasedParametrizer
):
    testing_data = AlignerAlgorithmTestData()


@pytest.mark.xfail
class TestGraphAlignerAlgorithmCmp(
    AlignerAlgorithmCmpTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 4)
    total_space = GraphSpace(_n)
    total_space.equip_with_group_action()

    aligner = ExhaustiveAligner(total_space)
    other_aligner = FAQAligner(total_space)

    testing_data = GraphAlignerCmpTestData()


@pytest.fixture(
    scope="class",
    params=[
        (PointToGeodesicAligner, dict(s_min=0.0, s_max=1.0, n_grid=100)),
        (_GeodesicToPointAligner, dict()),
    ],
)
def point_to_geodesic_aligners(request):
    Aligner, kwargs = request.param

    _n = random.randint(2, 4)
    request.cls.total_space = total_space = GraphSpace(_n)
    total_space.equip_with_group_action()
    total_space.equip_with_quotient()

    request.cls.aligner = Aligner(total_space, **kwargs)

    request.cls.data_generator = RandomDataGenerator(total_space)


@pytest.mark.usefixtures("point_to_geodesic_aligners")
class TestPointToGeodesicAligner(
    PointToGeodesicAlignerTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PointToGeodesicAlignerTestData()


class TestGraphSpace(MatricesTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = GraphSpace(_n, equip=False)
    testing_data = MatricesTestData()


class TestGraphSpaceQuotientMetric(
    QuotientMetricWithArrayTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 4)
    total_space = GraphSpace(_n)
    total_space.equip_with_group_action()
    total_space.equip_with_quotient()

    space = total_space.quotient

    testing_data = QuotientMetricWithArrayTestData()
