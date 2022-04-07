"""Unit tests for the graphspace quotient space."""

import geomstats.backend as gs
from geomstats.geometry.graphspace import GraphSpace, GraphSpaceMetric
from tests.conftest import Parametrizer, TestCase
from tests.data_generation import TestData


class TestGraphSpace(TestCase, metaclass=Parametrizer):
    space = GraphSpace

    class GraphSpaceTestData(TestData):
        def belongs_test_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    mat=gs.array(
                        [[[3.0, -1.0], [-1.0, 3.0]], [[4.0, -6.0], [-1.0, 3.0]]]
                    ),
                    expected=[True, True],
                ),
                dict(dim=2, mat=gs.array([-1.0, -1.0]), expected=False),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_test_data(self):
            smoke_data = [dict(n=2, n_points=1), dict(n=2, n_points=10)]
            return self.generate_tests(smoke_data)

        def permute_test_data(self):
            smoke_data = [
                dict(
                    n=2,
                    graph=gs.array([[0.0, 1.0], [2.0, 3.0]]),
                    permutation=[1, 0],
                    expected=gs.array([[3.0, 2.0], [1.0, 0.0]]),
                )
            ]
            return self.generate_tests(smoke_data)

    testing_data = GraphSpaceTestData()

    def test_random_point_belongs(self, n, n_points):
        space = self.space(n)
        point = space.random_point(n_points)
        result = gs.all(space.belongs(point))
        self.assertAllClose(result, True)

    def test_belongs(self, n, mat, expected):
        space = self.space(n)
        self.assertAllClose(space.belongs(gs.array(mat)), gs.array(expected))

    def test_permute(self, n, graph, permutation, expected):
        space = self.space(n)
        result = space.permute(gs.array(graph), permutation)
        self.assertAllClose(result, expected)


class TestGraphSpaceMetric(TestCase, metaclass=Parametrizer):
    metric = GraphSpaceMetric

    class GraphSpaceMetricTestData(TestData):
        def matchers_test_data(self):
            smoke_data = [
                dict(
                    n=2,
                    set1=gs.array([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 0.0], [0.0, 1.0]]]),
                    set2=gs.array([[[3.0, 2.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]),
                )
            ]

            return self.generate_tests(smoke_data)

    testing_data = GraphSpaceMetricTestData()

    def test_matchers(self, n, set1, set2):
        metric = self.metric(n)
        d1 = metric.dist(set1, set2, matcher="FAQ")
        d2 = metric.dist(set1, set2, matcher="ID")
        result1 = d1[0] < d2[0]
        result2 = d1[1] == d2[1]
        self.assertAllClose(result1, True)
        self.assertAllClose(result2, True)
