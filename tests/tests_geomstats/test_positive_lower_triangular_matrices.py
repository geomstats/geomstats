"""Unit tests for Positive lower triangular matrices"""


import math

import geomstats.backend as gs
from geomstats.geometry.positive_lower_triangular_matrices import (
    CholeskyMetric,
    PositiveLowerTriangularMatrices,
)
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.conftest import Parametrizer, TestCase, TestData

EULER = gs.exp(1.0)
SQRT_2 = math.sqrt(2)


class TestPositiveLowerTriangularMatrices(TestCase, metaclass=Parametrizer):
    """Test of Cholesky methods."""

    space = PositiveLowerTriangularMatrices

    class TestDataPositiveLowerTriangularMatrices(TestData):
        def belongs_data(self):
            smoke_data = [
                dict(n=2, mat=[[1.0, 0.0], [-1.0, 3.0]], expected=True),
                dict(n=2, mat=[[1.0, -1.0], [-1.0, 3.0]], expected=False),
                dict(n=2, mat=[[-1.0, 0.0], [-1.0, 3.0]], expected=False),
                dict(n=3, mat=[[1.0, 0], [0, 1.0]], expected=False),
                dict(
                    n=2,
                    mat=[
                        [[1.0, 0], [0, 1.0]],
                        [[1.0, 2.0], [2.0, 1.0]],
                        [[-1.0, 0.0], [1.0, 1.0]],
                        [[0.0, 0.0], [1.0, 1.0]],
                    ],
                    expected=[True, False, False, False],
                ),
                dict(
                    n=3,
                    mat=[
                        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[-1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    ],
                    expected=[False, False, True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def random_point_and_belongs_data(self):
            smoke_data = [
                dict(n=1, n_samples=1),
                dict(n=2, n_samples=2),
                dict(n=10, n_samples=100),
                dict(n=100, n_samples=10),
            ]
            return self.generate_tests(smoke_data)

        def gram_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point=[[1.0, 0.0], [2.0, 1.0]],
                    expected=[[1.0, 2.0], [2.0, 5.0]],
                ),
                dict(
                    n=2,
                    point=[[[2.0, 1.0], [0.0, 1.0]], [[-6.0, 0.0], [5.0, 3.0]]],
                    expected=[[[5.0, 1.0], [1.0, 1.0]], [[36.0, -30.0], [-30.0, 34.0]]],
                ),
            ]
            return self.generate_tests(smoke_data)

        def differential_gram_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec=[[-1.0, 0.0], [2.0, -1.0]],
                    base_point=[[1.0, 0.0], [2.0, 1.0]],
                    expected=[[-2.0, 0.0], [0.0, 6.0]],
                ),
                dict(
                    n=2,
                    tangent_vec=[[[-1.0, 2.0], [2.0, -1.0]], [[0.0, 4.0], [4.0, -1.0]]],
                    base_point=[[[3.0, 0.0], [-1.0, 2.0]], [[4.0, 0.0], [-1.0, 4.0]]],
                    expected=[
                        [[-6.0, 11.0], [11.0, -8.0]],
                        [[0.0, 32.0], [32.0, -16.0]],
                    ],
                ),
            ]
            return self.generate_tests(smoke_data)

        def inverse_differential_gram_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec=[[1.0, 2.0], [2.0, 5.0]],
                    base_point=[[1.0, 0.0], [2.0, 2.0]],
                    expected=[[0.5, 0.0], [1.0, 0.25]],
                ),
                dict(
                    n=2,
                    tangent_vec=[[[-4.0, 1.0], [1.0, -4.0]], [[0.0, 4.0], [4.0, -8.0]]],
                    base_point=[[[2.0, 0.0], [-1.0, 2.0]], [[4.0, 0.0], [-1.0, 2.0]]],
                    expected=[[[-1.0, 0.0], [0.0, -1.0]], [[0.0, 0.0], [1.0, -1.5]]],
                ),
            ]
            return self.generate_tests(smoke_data)

        def differential_gram_belongs_data(self):
            n_list = [1, 2, 2, 3, 10, 5, 100, 1000]
            n_samples_list = [1, 1, 2, 50, 1000, 100, 10, 5]
            space = PositiveLowerTriangularMatrices
            random_data = [
                dict(
                    n=n,
                    tangent_vec=space(n).ambient_space.random_point(n_samples),
                    base_point=space(n).random_point(n_samples),
                )
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def inverse_differential_gram_belongs_data(self):
            n_list = [1, 2, 2, 3, 10, 15, 100, 1000]
            n_samples_list = [1, 1, 2, 200, 1000, 100, 10, 5]
            space = PositiveLowerTriangularMatrices
            random_data = [
                dict(
                    n=n,
                    tangent_vec=space(n).ambient_space.random_point(n_samples),
                    base_point=space(n).random_point(n_samples),
                )
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

    testing_data = TestDataPositiveLowerTriangularMatrices()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(self.space(n).belongs(gs.array(mat)), gs.array(expected))

    def test_random_point_and_belongs(self, n, n_samples):
        space_n = self.space(n)
        self.assertAllClose(
            gs.all(space_n.belongs(space_n.random_point(n_samples))), True
        )

    def test_gram(self, n, point, expected):
        self.assertAllClose(self.space(n).gram(gs.array(point)), gs.array(expected))

    def test_differential_gram(self, n, tangent_vec, base_point, expected):
        self.assertAllClose(
            self.space(n).differential_gram(
                gs.array(tangent_vec), gs.array(base_point)
            ),
            gs.array(expected),
        )

    def test_inverse_differential_gram(self, n, tangent_vec, base_point, expected):
        self.assertAllClose(
            self.space(n).inverse_differential_gram(
                gs.array(tangent_vec), gs.array(base_point)
            ),
            gs.array(expected),
        )

    def test_differential_gram_belongs(self, n, tangent_vec, base_point):
        result = self.space(n).differential_gram(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(gs.all(SymmetricMatrices(n).belongs(result)), True)

    def test_inverse_differential_gram_belongs(self, n, tangent_vec, base_point):
        result = self.space(n).inverse_differential_gram(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(gs.all(self.space(n).ambient_space.belongs(result)), True)


class TestCholeskyMetric(TestCase, metaclass=Parametrizer):
    cls = CholeskyMetric
    space = PositiveLowerTriangularMatrices

    class TestDataCholeskyMetric(TestData):
        def diag_inner_product_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec_a=[[1.0, 0.0], [-2.0, -1.0]],
                    tangent_vec_b=[[2.0, 0.0], [-3.0, -1.0]],
                    base_point=[[SQRT_2, 0.0], [-3.0, 1.0]],
                    expected=2.0,
                )
            ]
            return self.generate_tests(smoke_data)

        def strictly_lower_inner_product_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec_a=[[1.0, 0.0], [-2.0, -1.0]],
                    tangent_vec_b=[[2.0, 0.0], [-3.0, -1.0]],
                    expected=6.0,
                )
            ]
            return self.generate_tests(smoke_data)

        def inner_product_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec_a=[[1.0, 0.0], [-2.0, -1.0]],
                    tangent_vec_b=[[2.0, 0.0], [-3.0, -1.0]],
                    base_point=[[SQRT_2, 0.0], [-3.0, 1.0]],
                    expected=8.0,
                )
            ]
            return self.generate_tests(smoke_data)

        def exp_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec=[[-1.0, 0.0], [2.0, 3.0]],
                    base_point=[[1.0, 0.0], [2.0, 2.0]],
                    expected=[[1 / EULER, 0.0], [4.0, 2 * gs.exp(1.5)]],
                )
            ]
            return self.generate_tests(smoke_data)

        def log_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point=[[EULER, 0.0], [2.0, EULER ** 3]],
                    base_point=[[EULER ** 3, 0.0], [4.0, EULER ** 4]],
                    expected=[[-2.0 * EULER ** 3, 0.0], [-2.0, -1 * EULER ** 4]],
                )
            ]
            return self.generate_tests(smoke_data)

        def squared_dist_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point_a=[[EULER, 0.0], [2.0, EULER ** 3]],
                    point_b=[[EULER ** 3, 0.0], [4.0, EULER ** 4]],
                    expected=9,
                ),
                dict(
                    n=2,
                    point_a=[
                        [[EULER, 0.0], [2.0, EULER ** 3]],
                        [[EULER, 0.0], [4.0, EULER ** 3]],
                    ],
                    point_b=[
                        [[EULER ** 3, 0.0], [4.0, EULER ** 4]],
                        [[EULER ** 3, 0.0], [7.0, EULER ** 4]],
                    ],
                    expected=[9, 14],
                ),
            ]
            return self.generate_tests(smoke_data)

        def exp_belongs_data(self):
            n_list = [1, 2, 2, 3, 10, 100, 1000]
            n_points_list = [1, 1, 2, 1000, 100, 10, 5]
            space = PositiveLowerTriangularMatrices
            random_data = [
                dict(
                    n=n,
                    tangent_vec=space(n).ambient_space.random_point(n_point),
                    base_point=space(n).random_point(n_point),
                )
                for n, n_point in zip(n_list, n_points_list)
            ]
            return self.generate_tests([], random_data)

        def log_belongs_data(self):
            n_list = [1, 2, 2, 3, 10, 100, 1000]
            n_samples_list = [1, 1, 2, 1000, 100, 10, 5]
            space = PositiveLowerTriangularMatrices
            random_data = [
                dict(
                    n=n,
                    point=space(n).random_point(n_samples),
                    base_point=space(n).random_point(n_samples),
                )
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

    testing_data = TestDataCholeskyMetric()

    def test_diag_inner_product(
        self, n, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        result = self.cls(n).diag_inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_strictly_lower_inner_product(
        self, n, tangent_vec_a, tangent_vec_b, expected
    ):
        result = self.cls(n).strictly_lower_inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_inner_product(self, n, tangent_vec_a, tangent_vec_b, base_point, expected):
        result = self.cls(n).inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_exp(self, n, tangent_vec, base_point, expected):
        result = self.cls(n).exp(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_log(self, n, point, base_point, expected):
        result = self.cls(n).log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_squared_dist(self, n, point_a, point_b, expected):
        result = self.cls(n).squared_dist(gs.array(point_a), gs.array(point_b))
        self.assertAllClose(result, gs.array(expected))

    def test_exp_belongs(self, n, tangent_vec, base_point):
        result = self.space(n).belongs(
            self.cls(n).exp(gs.array(tangent_vec), gs.array(base_point))
        )
        self.assertAllClose(gs.all(result), True)

    def test_log_belongs(self, n, point, base_point):
        result = self.space(n).ambient_space.belongs(
            self.cls(n).log(gs.array(point), gs.array(base_point))
        )
        self.assertAllClose(gs.all(result), True)
