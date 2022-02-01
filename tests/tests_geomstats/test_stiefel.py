"""Unit tests for Stiefel manifolds."""

import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.stiefel import Stiefel, StiefelCanonicalMetric
from tests.conftest import MetricParametrizer, Parametrizer, TestCase, TestData

p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
r_z = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
point1 = gs.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

point_a = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

point_b = gs.array(
    [
        [1.0 / gs.sqrt(2.0), 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0 / gs.sqrt(2.0), 0.0, 0.0],
    ]
)

point_perp = gs.array([[0.0], [0.0], [0.0], [1.0]])

matrix_a_1 = gs.array([[0.0, 2.0, -5.0], [-2.0, 0.0, -1.0], [5.0, 1.0, 0.0]])

matrix_b_1 = gs.array([[-2.0, 1.0, 4.0]])

matrix_a_2 = gs.array([[0.0, 2.0, -5.0], [-2.0, 0.0, -1.0], [5.0, 1.0, 0.0]])

matrix_b_2 = gs.array([[-2.0, 1.0, 4.0]])

tangent_vec_a = gs.matmul(point_a, matrix_a_1) + gs.matmul(point_perp, matrix_b_1)

tangent_vec_b = gs.matmul(point_a, matrix_a_2) + gs.matmul(point_perp, matrix_b_2)


class TestStiefel(TestCase, metaclass=Parametrizer):
    cls = Stiefel

    class TestDataStiefel(TestData):
        def random_point_belongs_shape_data(self):
            random_data = []
            n_list = random.sample(range(2, 50), 5)
            for n in n_list:
                p_list = random.sample(range(1, n), 5)
                for p in p_list:
                    n_samples = random.sample(range(2, 50))
                    random_data += [dict(n=n, p=p, n_samples=n_samples)]

            return self.generate_tests([], random_data)

        def to_grassmannian(self):

            point1 = gs.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]) / gs.sqrt(2.0)
            batch_points = Matrices.mul(
                GeneralLinear.exp(gs.array([gs.pi * r_z / n for n in [2, 3, 4]])),
                point1,
            )
            smoke_data = [
                dict(n=2, p=3, point=point1, expected=p_xy),
                dict(
                    n=2, p=3, point=batch_points, expected=gs.array([p_xy, p_xy, p_xy])
                ),
            ]
            return self.generate_tests(smoke_data)

    def test_random_point_belongs_shape(self, n, p, n_samples):
        result = gs.all(self.cls(n, p).belongs(self.cls(n, p).random_point(n_samples)))
        self.assertAllClose(result, gs.array(True))

    def test_to_tangent_is_tangent(self, n, p, n_samples):
        space = self.cls(n, p)
        point = space.random_point(n_samples)
        vector = gs.random.rand(*point.shape) / 4
        tangent_vec = self.space.to_tangent(vector, point)
        result = self.space.is_tangent(tangent_vec, point)
        self.assertAllClose(result, gs.array(True))

    def test_projection_belongs(self, n, p, n_samples):
        space = self.cls(n, p)
        shape = (n_samples, n, p)
        belongs = gs.all(space.belongs(space.projection(gs.random.normal(size=shape))))
        self.assertAllClose(belongs, gs.array(True))

    def test_to_grassmannian(self, n, p, point, expected):
        self.assertAllClose(
            self.cls(n, p).to_grassmannian(gs.array(point)), gs.array(expected)
        )


class TestStiefelCanonicalMetric(TestCase, metaclass=MetricParametrizer):
    cls = StiefelCanonicalMetric
    space = Stiefel

    class TestDataStiefelCanonicalMetric(TestData):
        def log_two_sheets_error(self):
            stiefel = Stiefel(n=3, p=3)
            base_point = stiefel.random_point()
            det_base = gs.linalg.det(base_point)
            point = stiefel.random_point()
            det_point = gs.linalg.det(point)
            if gs.all(det_base * det_point > 0.0):
                point *= -1.0

            random_data = [
                dict(
                    n=3,
                    p=3,
                    point=point,
                    base_point=base_point,
                    expected=pytest.raises(ValueError),
                )
            ]
            return self.generate_tests([], random_data)

        def inner_product_shape_data(self):
            smoke_data = [
                dict(
                    n=2,
                    p=3,
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=point_a,
                )
            ]
            return self.generate_tests(smoke_data)

    def test_log_two_sheets_error(self, n, p, point, base_point, expected):
        metric = self.space(n, p)
        with expected:
            metric.log(point, base_point)

    def test_inner_product(self, n, p, tangent_vec_a, tangent_vec_b, base_point):
        metric = self.space(n, p)
        exp = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(gs.shape(exp), ())
