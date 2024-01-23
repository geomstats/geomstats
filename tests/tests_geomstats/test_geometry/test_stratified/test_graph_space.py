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

from .data.graph_space import (
    GraphAlignerCmpTestData,
    GraphSpaceQuotientMetricTestData,
    GraphSpaceTestData,
    PointToGeodesicAlignerTestData,
)
from .data.quotient import AlignerAlgorithmTestData


@pytest.fixture(
    scope="class",
    params=[
        ExhaustiveAligner,
        FAQAligner,
    ],
)
def aligner_algorithms(request):
    Aligner = request.param
    request.cls.aligner = Aligner()


@pytest.mark.usefixtures("aligner_algorithms")
class TestGraphAlignerAlgorithm(
    AlignerAlgorithmTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 4)
    total_space = GraphSpace(_n)
    total_space.equip_with_group_action()

    testing_data = AlignerAlgorithmTestData()


class TestGraphAlignerAlgorithmCmp(
    AlignerAlgorithmCmpTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 4)
    total_space = GraphSpace(_n)
    total_space.equip_with_group_action()

    aligner = ExhaustiveAligner()
    other_aligner = FAQAligner()

    testing_data = GraphAlignerCmpTestData()


@pytest.fixture(
    scope="class",
    params=[
        PointToGeodesicAligner(s_min=0.0, s_max=1.0),
        _GeodesicToPointAligner(),
    ],
)
def point_to_geodesic_aligners(request):
    request.cls.aligner = request.param


@pytest.mark.usefixtures("point_to_geodesic_aligners")
class TestPointToGeodesicAligner(
    PointToGeodesicAlignerTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 4)
    total_space = GraphSpace(_n)
    total_space.equip_with_group_action()
    total_space.equip_with_quotient_structure()

    data_generator = RandomDataGenerator(total_space)
    testing_data = PointToGeodesicAlignerTestData()


class TestGraphSpace(MatricesTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = GraphSpace(_n, equip=False)
    testing_data = GraphSpaceTestData()


class TestGraphSpaceQuotientMetric(
    QuotientMetricWithArrayTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 4)
    total_space = GraphSpace(_n)
    total_space.equip_with_group_action()
    total_space.equip_with_quotient_structure()

    space = total_space.quotient

    testing_data = GraphSpaceQuotientMetricTestData()
