"""Unit tests for the graphspace quotient space."""

import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.stratified.graph_space import (
    ExhaustiveAligner,
    FAQAligner,
    GraphSpace,
    IDAligner,
    PointToGeodesicAligner,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.matrices import MatricesTestCase
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
        IDAligner,
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


class TestPointToGeodesicAligner(TestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 4)
    total_space = GraphSpace(_n)
    total_space.equip_with_group_action()
    total_space.equip_with_quotient_structure()

    aligner = PointToGeodesicAligner(s_min=0.0, s_max=1.0)

    data_generator = RandomDataGenerator(total_space)
    testing_data = PointToGeodesicAlignerTestData()

    def test_align(self, geodesic, point, expected, atol):
        res = self.aligner.align(self.total_space, geodesic, point)
        self.assertAllClose(res, expected, atol=atol)

    def test_align_with_endpoints(
        self, initial_point, end_point, point, expected, atol
    ):
        geodesic = self.total_space.quotient.metric.geodesic(initial_point, end_point)
        self.test_align(geodesic, point, expected, atol)

    @pytest.mark.vec
    def test_align_vec(self, n_reps, atol):
        initial_point = self.data_generator.random_point()
        end_point = self.data_generator.random_point()

        geodesic = self.total_space.quotient.metric.geodesic(initial_point, end_point)
        point = self.data_generator.random_point()

        expected = self.aligner.align(self.total_space, geodesic, point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    initial_point=initial_point,
                    end_point=end_point,
                    point=point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["initial_point", "end_point", "point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data, test_fnc_name="test_align_with_endpoints")

    def test_dist(self, geodesic, point, expected, atol):
        res = self.aligner.dist(self.total_space, geodesic, point)
        self.assertAllClose(res, expected, atol=atol)

    def test_dist_with_endpoints(self, initial_point, end_point, point, expected, atol):
        geodesic = self.total_space.quotient.metric.geodesic(initial_point, end_point)
        self.test_dist(geodesic, point, expected, atol)

    @pytest.mark.vec
    def test_dist_vec(self, n_reps, atol):
        initial_point = self.data_generator.random_point()
        end_point = self.data_generator.random_point()

        geodesic = self.total_space.quotient.metric.geodesic(initial_point, end_point)
        point = self.data_generator.random_point()

        expected = self.aligner.dist(self.total_space, geodesic, point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    initial_point=initial_point,
                    end_point=end_point,
                    point=point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["initial_point", "end_point", "point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data, test_fnc_name="test_dist_with_endpoints")

    @pytest.mark.random
    def test_dist_along_geodesic_is_zero(self, n_points, atol):
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        geodesic = self.total_space.quotient.metric.geodesic(
            initial_point,
            end_point,
        )

        s = gs.random.rand(1)
        points = gs.squeeze(geodesic(s), axis=-3)

        batch_shape = (n_points,) if n_points > 1 else ()
        self.test_dist_with_endpoints(
            initial_point, end_point, points, gs.zeros(batch_shape), atol
        )


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
