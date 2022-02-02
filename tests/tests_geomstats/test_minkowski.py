"""Unit tests for Minkowski space."""

import math

import geomstats.backend as gs
from geomstats.geometry.minkowski import Minkowski, MinkowskiMetric
from tests.conftest import MetricParametrizer, Parametrizer, TestCase, TestData


class TestMinkowski(TestCase, metaclass=Parametrizer):
    cls = Minkowski

    class TestDataMinkowski(TestData):
        def belongs_data(self):
            smoke_data = [dict(dim=2, point=[-1.0, 3.0], expected=True)]
            return self.generate_tests(smoke_data)

    def test_belongs(self, dim, point, expected):
        self.assertAllClose(self.cls(dim).belongs(gs.array(point)), gs.array(expected))


class TestMinkowskiMetric(TestCase, metaclass=MetricParametrizer):
    cls = MinkowskiMetric

    class TestDataMinkowskiMetric(TestData):
        def metric_matrix_Data(self):
            smoke_data = [dict(dim=2, expected=[[-1.0, 0.0], [0.0, 1.0]])]
            return self.generate_tests(smoke_data)

        def inner_product(self):
            smoke_data = [
                dict(dim=2, point_a=[0.0, 1.0], point_b=[2.0, 10.0], expected=10.0),
                dict(
                    dim=2,
                    point_a=[[-1.0, 0.0], [1.0, 0.0], [2.0, math.sqrt(3)]],
                    point_b=[
                        [2.0, -math.sqrt(3)],
                        [4.0, math.sqrt(15)],
                        [-4.0, math.sqrt(15)],
                    ],
                    expected=[2.0, -4.0, 14.70820393],
                ),
            ]
            return self.generate_tests(smoke_data)

        def squared_norm_data(self):
            pass

    def test_metric_matrix(self, dim, expected):
        metric = self.cls(dim)
        self.assertAllClose(metric.metric_matrix(), gs.array(expected))

    def test_inner_product(self, dim, point_a, point_b, expected):
        metric = self.cls(dim)
        self.assertAllClose(
            metric.inner_product(gs.array(point_a), gs.array(point_b)),
            gs.array(expected),
        )

    def test_squared_norm(self, dim, point, expected):
        metric = self.cls(dim)
        self.assertAllClose(metric.squared_norm(gs.array(point), gs.array(expected)))
