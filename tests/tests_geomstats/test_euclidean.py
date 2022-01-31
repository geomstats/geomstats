"""Unit tests for the Euclidean space."""

import itertools
import random

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from tests.conftest import MetricParametrizer, Parametrizer, TestCase, TestData

SQRT_2 = gs.sqrt(2)
SQRT_5 = gs.sqrt(5)


class TestEuclidean(TestCase, metaclass=Parametrizer):
    cls = Euclidean

    class TestDataEuclidean(TestData):
        def belongs_data(self):
            smoke_data = [
                dict(dim=2, vec=[0.0, 1.0], expected=True),
                dict(dim=2, vec=[1.0, 0.0, 1.0], expected=False),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_data(self):
            dim_list = random.sample(range(1, 100), 10)
            n_samples_list = random.sample(range(1, 100), 10)
            smoke_data = [
                dict(dim=1, n_samples=1),
                dict(dim=2, n_samples=1),
                dict(dim=3, n_samples=10),
            ]
            random_data = [
                dict(dim=dim, n_samples=n_samples)
                for dim, n_samples in zip(dim_list, n_samples_list)
            ]

            return self.generate_tests(smoke_data, random_data)

        def random_point_is_tangent_data(self):
            dim_list = random.sample(range(1, 100), 10)
            n_samples_list = random.sample(range(1, 100), 10)
            smoke_data = [
                dict(dim=1, n_samples=1),
                dict(dim=2, n_samples=1),
                dict(dim=3, n_samples=10),
            ]
            random_data = [
                dict(dim=dim, n_samples=n_samples)
                for dim, n_samples in zip(dim_list, n_samples_list)
            ]

            return self.generate_tests(smoke_data, random_data)

        def to_tangent_is_identity_data(self):
            dim_list = random.sample(range(1, 100), 10)
            n_samples_list = random.sample(range(1, 100), 10)
            smoke_data = [
                dict(dim=1, n_samples=1),
                dict(dim=2, n_samples=1),
                dict(dim=3, n_samples=10),
            ]
            random_data = [
                dict(dim=dim, n_samples=n_samples)
                for dim, n_samples in zip(dim_list, n_samples_list)
            ]

            return self.generate_tests(smoke_data, random_data)

    testing_data = TestDataEuclidean()

    def test_belongs(self, dim, vec, expected):
        self.assertAllClose(self.cls(dim).belongs(gs.array(vec)), gs.array(expected))

    def test_random_point_belongs(self, dim, n_samples):
        points = self.cls(dim).random_point(n_samples)
        self.assertAllClose(gs.all(self.cls(dim).belongs(points)), gs.array(True))

    def test_random_point_is_tangent(self, dim, n_samples):
        points = self.cls(dim).random_point(n_samples)
        self.assertAllClose(gs.all(self.cls(dim).is_tangent(points)), gs.array(True))

    def test_random_point_is_identity(self, dim, n_samples):
        points = self.cls(dim).random_point(n_samples)
        self.assertAllClose(self.cls(dim).to_tangent(points), points)


class TestEuclideanMetric(TestCase, metaclass=MetricParametrizer):
    cls = EuclideanMetric
    space = Euclidean

    class TestDataEuclideanMetric(TestData):
        def exp_data(self):

            one_tv = gs.array([0.0, 1.0])
            one_bp = gs.array([2.0, 10.0])
            n_tvs = gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]])
            n_bps = gs.array([[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]])
            smoke_data = [
                dict(
                    dim=2,
                    tangent_vec=[0.0, 1.0],
                    base_point=[2.0, 10.0],
                    expected=[2.0, 11.0],
                ),
                dict(
                    dim=2,
                    tangent_vec=one_tv,
                    base_point=one_bp,
                    expected=one_tv + one_bp,
                ),
                dict(
                    dim=2, tangent_vec=one_tv, base_point=n_bps, expected=one_tv + n_bps
                ),
                dict(
                    dim=2, tangent_vec=n_tvs, base_point=one_bp, expected=n_tvs + one_bp
                ),
                dict(
                    dim=2, tangent_vec=n_tvs, base_point=n_bps, expected=n_tvs + n_bps
                ),
            ]
            return self.generate_tests(smoke_data)

        def log_data(self):
            one_p = gs.array([0.0, 1.0])
            one_bp = gs.array([2.0, 10.0])
            n_ps = gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]])
            n_bps = gs.array([[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]])
            smoke_data = [
                dict(
                    dim=2, point=[2.0, 10.0], base_point=[0.0, 1.0], expected=[2.0, 9.0]
                ),
                dict(dim=2, point=one_p, base_point=one_bp, expected=one_p - one_bp),
                dict(dim=2, point=one_p, base_point=n_bps, expected=one_p - n_bps),
                dict(dim=2, point=n_ps, base_point=one_bp, expected=n_ps - one_bp),
                dict(dim=2, point=n_ps, base_point=n_bps, expected=n_ps - n_bps),
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
                    vec=[0.0, 1.0],
                    expected=1.0,
                ),
                dict(
                    dim=2,
                    vec=[[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]],
                    expected=[5.0, 20.0, 26.0],
                ),
            ]
            return self.generate_tests(smoke_data)

        def norm_data(self):
            smoke_data = [
                dict(dim=2, vec=[4.0, 3.0], expected=5.0),
                dict(dim=4, vec=[4.0, 3.0, 4.0, 3.0], expected=5.0 * SQRT_2),
                dict(
                    dim=3,
                    vec=[[4.0, 3.0, 10.0], [3.0, 10.0, 4.0]],
                    expected=[5 * SQRT_5, 5 * SQRT_5],
                ),
            ]
            return self.generate_tests(smoke_data)

        def metric_matrix_data(self):
            smoke_data = [
                dict(dim=1, expected=gs.eye(1)),
                dict(dim=2, expected=gs.eye(2)),
                dict(dim=3, expected=gs.eye(3)),
            ]
            return self.generate_tests(smoke_data)

        def log_exp_composition_data(self):
            dim_list = random.sample(range(2, 50), 10)
            args = [
                (metric_args, Euclidean(metric_args[0]))
                for metric_args in itertools.product(dim_list)
            ]
            return self._log_exp_composition_data(args)

        def exp_belongs_data(self):
            dim_list = random.sample(range(2, 50), 10)
            args = [
                (metric_args, Euclidean(metric_args[0]))
                for metric_args in itertools.product(dim_list)
            ]
            return self._exp_belongs_data(args)

        def log_is_tangent_data(self):
            dim_list = random.sample(range(2, 50), 10)
            args = [
                (metric_args, Euclidean(metric_args[0]))
                for metric_args in itertools.product(dim_list)
            ]
            return self._log_is_tangent_data(args)

        def squared_dist_is_symmetric_data(self):
            dim_list = random.sample(range(2, 50), 10)
            args = [
                (metric_args, Euclidean(metric_args[0]))
                for metric_args in itertools.product(dim_list)
            ]
            return self._squared_dist_is_symmetric_data(args)

    testing_data = TestDataEuclideanMetric()

    def test_exp(self, dim, tangent_vec, base_point, expected):
        metric = EuclideanMetric(dim)
        self.assertAllClose(
            metric.exp(gs.array(tangent_vec), gs.array(base_point)), gs.array(expected)
        )

    def test_log(self, dim, point, base_point, expected):
        metric = EuclideanMetric(dim)
        self.assertAllClose(
            metric.log(gs.array(point), gs.array(base_point)), gs.array(expected)
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

    def test_norm(self, dim, vec, expected):
        metric = EuclideanMetric(dim)
        self.assertAllClose(metric.norm(gs.array(vec)), gs.array(expected))

    def test_metric_matrix(self, dim, expected):
        self.assertAllClose(EuclideanMetric(dim).metric_matrix(), gs.array(expected))
