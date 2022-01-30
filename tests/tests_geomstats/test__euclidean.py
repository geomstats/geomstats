"""Unit tests for the Euclidean space."""

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from tests.conftest import Parametrizer, TestCase, TestData

SQRT_2 = gs.sqrt(2)
SQRT_5 = gs.sqrt(5)


class TestEuclidean(TestCase, metaclass=Parametrizer):
    class TestDataEuclidean(TestData):
        def belongs_data(self):
            smoke_data = [
                dict(dim=2, vec=[0.0, 1.0], expected=True),
                dict(dim=2, vec=[1.0, 0.0, 1.0], expected=False),
            ]
            return self.generate_tests(smoke_data)

    def test_belongs(self, dim, vec, expected):
        self.assertAllClose(gs.all(Euclidean(dim).belongs(vec)), expected)


class TestEuclideanMetric(TestCase, metaclass=Parametrizer):
    class TestDataEuclideanMetric(TestData):
        def exp_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    tangent_vec=[0.0, 1.0],
                    base_point=[2.0, 10.0],
                    expected=[2.0, 11.0],
                )
            ]
            return self.generate_tests(smoke_data)

        def log_data(self):
            smoke_data = [
                dict(
                    dim=2, base_point=[0.0, 1.0], point=[2.0, 10.0], expected=[2.0, 9.0]
                )
            ]
            return self.generate_tests(smoke_data)

        def inner_product_data(self):
            tangent_vec_1 = [[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]]
            tangent_vec_2 = [[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]]
            tangent_vec_3 = [0.0, 1.0]
            tangent_vec_4 = [2.0, 10.0]
            smoke_data = [
                dict(
                    dim=2,
                    tangent_vec_a=tangent_vec_1,
                    tangent_vec_b=tangent_vec_4,
                    expected=[14.0, -44.0, 0.0],
                ),
                dict(
                    dim=2,
                    tangent_vec_a=tangent_vec_3,
                    tangent_vec_b=tangent_vec_2,
                    expected=[10.0, -1.0, 6.0],
                ),
                dict(
                    dim=2,
                    tangent_vec_a=tangent_vec_1,
                    tangent_vec_b=tangent_vec_2,
                    expected=[14.0, -12.0, 21.0],
                ),
                dict(
                    dim=2,
                    tangent_vec_a=[0.0, 1.0],
                    tangent_vec_b=[2.0, 10.0],
                    expected=10.0,
                ),
            ]
            return self.generate_tests(smoke_data)

        def squared_norm_data(self):
            smoke_data = [
                dict(
                    dim=2,
                    vec=[[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]],
                    expected=[5.0, 20.0, 26.0],
                )
            ]
            return self.generate_tests(smoke_data)

        def norm_data(self):
            smoke_data = [
                dict(dim=2, vec=[4.0, 3.0], expected=5.0),
                dict(dim=4, vec=[4.0, 3.0, 4.0, 3.0], expected=5.0 * SQRT_2),
                dict(
                    dim=3,
                    vec=[[4.0, 3.0, 10.0], [3.0, 10.0, 4.0]],
                    expected=[SQRT_5, SQRT_5],
                ),
            ]
            return self.generate_tests(smoke_data)

    def test_exp(self, dim, tangent_vec, base_point, expected):
        metric = EuclideanMetric(dim)
        self.assertAllClose(
            metric.exp(gs.array(tangent_vec), gs.array(base_point)), gs.array(expected)
        )

    def test_log(self, dim, base_point, point, expected):
        metric = EuclideanMetric(dim)
        self.assertAllClose(
            metric.log(gs.array(base_point), gs.array(point), gs.array(expected))
        )

    def test_inner_product(self, dim, tangent_vec_a, tangent_vec_b, expected):
        metric = EuclideanMetric(dim)
        self.assertAllClose(
            metric.inner_product(gs.array(tangent_vec_a), gs.array(tangent_vec_b)),
            gs.array(expected),
        )

    def test_squared_norm(self, dim, vec, expected):
        metric = EuclideanMetric(dim)
        self.assertAllClose(metric.squared_norm(gs.array(vec)), gs.array(expected))

    def test_norm_data(self, dim, vec, expected):
        metric = EuclideanMetric(dim)
        self.assertAllClose(metric.norm(gs.array(vec)), gs.array(expected))
