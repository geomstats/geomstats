"""Unit tests for special euclidean group in matrix representation."""


import itertools
import random
from contextlib import nullcontext as does_not_raise

import pytest

import geomstats.backend as gs
from geomstats.geometry.special_euclidean import (
    SpecialEuclidean,
    SpecialEuclideanMatrixCannonicalLeftMetric,
    SpecialEuclideanMatrixLieAlgebra,
)
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.conftest import MetricParametrizer, Parametrizer, TestCase, TestData


def group_sample_matrix(theta):
    return [
        [gs.cos(theta), -gs.sin(theta), 2.0],
        [gs.sin(theta), gs.cos(theta), 3.0],
        [0.0, 0.0, 1.0],
    ]


def algebra_sample_matrix(theta):
    return [
        [gs.cos(theta), -gs.sin(theta), 2.0],
        [gs.sin(theta), gs.cos(theta), 3.0],
        [0.0, 0.0, 0.0],
    ]


class TestSpecialEuclidean(TestCase, metaclass=Parametrizer):

    cls = SpecialEuclidean

    class TestDataSpecialEuclidean(TestData):
        def belongs_data(self):
            smoke_data = [
                dict(n=2, mat=group_sample_matrix(gs.pi / 3), expected=True),
                dict(n=2, mat=algebra_sample_matrix(gs.pi / 3), expected=False),
                dict(
                    n=2,
                    mat=[
                        group_sample_matrix(gs.pi / 4),
                        algebra_sample_matrix(gs.pi / 6),
                    ],
                    expected=[True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_data(self):
            n_list = random.sample(range(2, 50), 10)
            smoke_data = [
                dict(n=2, n_samples=100),
                dict(n=3, n_samples=100),
                dict(n=10, n_samples=100),
            ]
            random_data = [dict(n=n, n_samples=100) for n in n_list]
            return self.generate_tests(smoke_data, random_data)

        def identity_data(self):
            smoke_data = [
                dict(n=2, expected=gs.eye(3)),
                dict(n=3, expected=gs.eye(4)),
                dict(n=10, expected=gs.eye(11)),
            ]
            return self.generate_tests(smoke_data)

        def is_tangent_data(self):
            theta = gs.pi / 3
            vec_1 = [[0.0, -theta, 2.0], [theta, 0.0, 3.0], [0.0, 0.0, 0.0]]
            vec_2 = [[0.0, -theta, 2.0], [theta, 0.0, 3.0], [0.0, 0.0, 0.0]]
            point = group_sample_matrix(theta)
            smoke_data = [
                dict(n=2, tangent_vec=point @ vec_1, base_point=point, expected=True),
                dict(n=2, tangent_vec=point @ vec_2, base_point=point, expected=False),
                dict(
                    n=2,
                    tangent_vec=[point @ vec_1, point @ vec_2],
                    base_point=point,
                    expected=[True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def compose_and_identity_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples = 100
            random_data = [
                dict(n=n, point=SpecialEuclidean(n).random_point(n_samples))
                for n in n_list
            ]
            for n in n_list:
                random_data.append()
            return self.generate_tests([], random_data)

        def basis_representation_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples = 100
            random_data = [
                dict(n=n, vec=gs.random.rand(n_samples, self.group.dim)) for n in n_list
            ]
            return self.generate_tests([], random_data)

        def test_metrics_default_point_type_data(self):
            n_list = random.sample(range(2, 50), 10)
            metric_str_list = [
                "left_canonical_metric",
                "right_canonical_metric",
                "metric",
            ]
            random_data = [arg for arg in itertools.product(n_list, metric_str_list)]
            return self.generate_tests([], random_data)

        def test_projection_belongs_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples = 100
            random_data = [
                dict(n=n, mat=gs.random.rand(n_samples, n + 1, n + 1)) for n in n_list
            ]
            return self.generate_tests([], random_data)

        def inverse_shape_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples = 10
            random_data = [
                dict(
                    n=n,
                    points=SpecialEuclidean(n).random_point(n_samples),
                    expected=(n_samples, n + 1, n + 1),
                )
                for n in n_list
            ]
            return self.generate_tests([], random_data)

        def compose_shape_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples = 10
            random_data = [
                dict(
                    n=n,
                    point_a=SpecialEuclidean(n).random_point(n_samples),
                    point_b=SpecialEuclidean(n).random_point(n_samples),
                    expected=(n_samples, n + 1, n + 1),
                )
                for n in n_list
            ]
            random_data += [
                dict(
                    n=n,
                    point_a=SpecialEuclidean(n).random_point(),
                    point_b=SpecialEuclidean(n).random_point(n_samples),
                    expected=(n_samples, n + 1, n + 1),
                )
                for n in n_list
            ]
            random_data += [
                dict(
                    n=n,
                    point_a=SpecialEuclidean(n).random_point(n_samples),
                    point_b=SpecialEuclidean(n).random_point(),
                    expected=(n_samples, n + 1, n + 1),
                )
                for n in n_list
            ]
            return self.generate_tests([], random_data)

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(SpecialEuclidean(n).belongs(mat), gs.array(expected))

    def test_random_point_belongs(self, n, n_samples):
        group = self.cls(n)
        self.assertAllClose(gs.all(group(n).random_point(n_samples)), gs.array(True))

    def test_identity(self, n, expected):
        self.assertAllClose(SpecialEuclidean(n).identity, gs.array(expected))

    def test_is_tangent(self, n, tangent_vec, base_point, expected):
        result = SpecialEuclidean(n).is_tangent(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_compose_and_identity(self, n, point):
        group = self.cls(n)
        result = group.compose(gs.array(point), group.inverse(gs.array(point)))
        self.assertAllClose(result, gs.broadcast_to(group.identity, result.shape))

    def test_basis_representation(self, n, point_type, vec):
        group = self.cls(n, point_type)
        tangent_vec = group.lie_algebra.matrix_representation(vec)
        result = group.lie_algebra.basis_representation(tangent_vec)
        self.assertAllClose(result, vec)

    def test_metrics_expected_point_type(self, n, point_type, metric_str):
        group = self.cls(n, point_type)
        self.assertTrue(getattr(group, metric_str).default_point_type == "matrix")

    def test_compose_identity_matrix(self, n, point_type, point):
        group = self.cls(n, point_type)
        self.assertAllClose(
            group.compose(gs.array(point), group.identity), gs.array(point)
        )

    def test_projection_belongs(self, n, point_type, mat):
        group = self.cls(n, point_type)
        self.assertAllClose(
            gs.all(group.belongs(group.projection(gs.array(mat)))), True
        )

    def test_inverse_shape_data(self, n, points, expected):
        group = self.cls(n)
        self.assertAllClose(gs.shape(group.inverse(points)), expected)

    def test_compose_shape_data(self, n, point_a, point_b, expected):
        group = self.cls(n)
        result = gs.shape(group.compose(gs.array(point_a), gs.array(point_b)))
        self.assertAllClose(result, expected)


class TestSpecialEuclideanMatrixLieAlgebra(TestCase, metaclass=Parametrizer):

    cls = SpecialEuclideanMatrixLieAlgebra

    class TestDataSpecialEuclideanMatrixLieAlgebra(TestData):
        def belongs_data(self):
            theta = gs.pi / 3
            smoke_data = [
                dict(n=2, vec=algebra_sample_matrix(theta), expected=True),
                dict(n=2, vec=group_sample_matrix(theta), expected=False),
                dict(
                    n=2,
                    vec=[algebra_sample_matrix(theta), group_sample_matrix(theta)],
                    expected=[True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def basis_belongs_data(self):
            n_list = random.sample(range(2, 50), 10)
            random_data = [dict(n=n) for n in n_list]
            return self.generate_tests([], random_data)

        def projection_belongs_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples = 100
            random_data = [
                dict(n=n, vec=gs.random.rand(n_samples, n + 1, n + 1)) for n in n_list
            ]
            return self.generate_tests([], random_data)

        def basis_dim_data(self):
            smoke_data = [
                dict(n=2, expected=3),
                dict(n=3, expected=6),
                dict(n=10, expected=55),
            ]
            return self.generate_tests(smoke_data)

        def basis_representation_shape_data(self):
            smoke_data = [
                dict(n=2, expected=(3, 3)),
                dict(n=3, expected=(6, 6)),
                dict(n=10, expected=(55, 55)),
            ]
            return self.generate_tests(smoke_data)

    def test_basis_belongs(self, n):
        algebra = self.cls(n)
        self.assertAllClose(gs.all(algebra.belongs(algebra.basis())), gs.array(True))

    def test_belongs(self, n, vec, expected):
        algebra = self.cls(n)
        self.assertAllClose(algebra.belongs(gs.array(vec)), gs.array(expected))

    def test_projection_belongs(self, n, vec):
        algebra = self.cls(n)
        self.assertAllClose(gs.all(algebra.belongs(algebra.projection(vec))), True)

    def test_basis_dim(self, n, expected):
        algebra = self.cls(n)
        self.assertAllClose(algebra.dim, expected)

    def test_basis_representation_shape(self, n, expected):
        algebra = self.cls(n)
        shape = gs.shape(algebra.basis_representation(algebra.basis))
        self.assertAllClose(shape, expected)


class TestSpecialEuclideanMatrixCannonicalLeftMetric(
    TestCase, metaclass=MetricParametrizer
):

    cls = SpecialEuclideanMatrixCannonicalLeftMetric
    space = SpecialEuclidean

    class TestDataSpecialEuclideanMatrixCannonicalLeftMetric(TestData):
        def left_metric_wrong_group_data(self):
            smoke_data = [
                dict(group=SpecialEuclidean(2), expected=does_not_raise()),
                dict(
                    group=SpecialEuclidean(2, point_type="vector"),
                    expected=pytest.raises(ValueError),
                ),
                dict(group=SpecialOrthogonal(3), expected=pytest.raises(ValueError)),
            ]
            return self.generate_tests(smoke_data)

        def log_exp_composition_data(self):
            n_list = random.sample(range(2, 50), 10)
            args = [
                (metric_args, SpecialEuclidean(metric_args[0]))
                for metric_args in itertools.product(n_list)
            ]
            return self._log_exp_composition_data(args)

        def exp_belongs_data(self):
            n_list = random.sample(range(2, 50), 10)
            args = [
                (metric_args, SpecialEuclidean(metric_args[0]))
                for metric_args in itertools.product(n_list)
            ]
            return self._exp_belongs_data(args)

        def log_is_tangent_data(self):
            n_list = random.sample(range(2, 50), 10)
            args = [
                (metric_args, SpecialEuclidean(metric_args[0]))
                for metric_args in itertools.product(n_list)
            ]
            return self._log_is_tangent_data(args)

        def squared_dist_is_symmetric_data(self):
            n_list = random.sample(range(2, 50), 10)
            args = [
                (metric_args, SpecialEuclidean(metric_args[0]))
                for metric_args in itertools.product(n_list)
            ]
            return self._squared_dist_is_symmetric_data(args)

    def test_left_metric_wrong_group(self, group, expected):
        with expected:
            self.cls(group)
