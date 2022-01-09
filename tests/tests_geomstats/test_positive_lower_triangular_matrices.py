"""Unit tests for Positive lower triangular matrices"""


import math

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.positive_lower_triangular_matrices import (
    CholeskyMetric,
    PositiveLowerTriangularMatrices,
)
from geomstats.geometry.spd_matrices import SPDMatrices
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
            return self.belongs_data(smoke_data)

        def random_point_and_belongs_data(self):
            smoke_data = [
                dict(n=1, n_points=1),
                dict(n=2, n_points=2),
                dict(n=10, n_points=100),
                dict(n=100, n_points=10),
            ]
            return self.generate_tests(smoke_data)

        def gram_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point=[[1.0, 0.0], [2.0, 1.0]],
                    expected=[[1.0, 2.0], [2.0, 5.0]],
                )
            ]
            return self.generate_tests(smoke_data)

        def differential_gram_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec=[[-1.0, 0.0], [2.0, -1.0]],
                    base_point=[[1.0, 0.0], [2.0, 1.0]],
                )
            ]
            return self.generate_tests(smoke_data)

        def inverse_differential_gram_data(self):
            smoke_data = [
                dict(
                    n=2,
                    tangent_vec=[[1.0, 2.0], [2.0, 5.0]],
                    base_point=[[1.0, 0.0], [2.0, 2.0]],
                    expected=[[0.5, 0.0], [1.0, 0.25]],
                )
            ]
            return self.generate_tests(smoke_data)

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(self.space(n).belongs(gs.array(mat)), gs.array(expected))

    def test_random_point_and_belongs(self, n, n_points):
        self.assertAllClose(gs.all(self.space(n).random_point(n_points)), True)

    def test_gram(self, n, point, expected):
        self.assertAllClose(self.space(n).gram(gs.array(point), gs.array(expected)))

    def test_differential_gram(self, n, tangent_vec, base_point):
        self.assertAllClose(
            self.space(n).differential_gram(gs.array(tangent_vec), gs.array(base_point))
        )

    def test_inverse_differential_gram(self, n, tangent_vec, base_point):
        self.assertAllClose(
            self.space(n).inverse_differential_gram(tangent_vec, base_point)
        )

    def test_gram_vectorization(self):
        point = self.space.random_point(n_samples=5)
        gram = self.space.gram(point)

        gram_result = gram
        gram_expected = gs.matmul(point, Matrices.transpose(point))
        self.assertAllClose(gram_expected, gram_result)

        belongs_result = gs.all(SPDMatrices(self.n).belongs(gram))
        belongs_expected = True
        self.assertAllClose(belongs_expected, belongs_result)

    def test_differential_gram_vectorization(self):
        base_point = self.space.random_point(self.n_samples)
        tangent_vec = self.space.ambient_space.random_point(self.n_samples)

        diff1 = gs.matmul(base_point, Matrices.transpose(tangent_vec))
        diff2 = gs.matmul(tangent_vec, Matrices.transpose(base_point))
        diff = diff1 + diff2

        belongs_result = gs.all(SymmetricMatrices(self.n).belongs(diff))
        belongs_expected = True
        self.assertAllClose(belongs_expected, belongs_result)

        diff_expected = diff
        diff_result = self.space.differential_gram(tangent_vec, base_point)
        self.assertAllClose(diff_expected, diff_result)

    def test_inv_differential_gram_belongs(self):
        L = self.space.random_point(5)
        W = SymmetricMatrices(2).random_point(5)
        inv_diff_gram_result = self.space.inverse_differential_gram(W, L)
        belongs_result = gs.all(self.space.ambient_space.belongs(inv_diff_gram_result))
        belongs_expected = True
        self.assertAllClose(belongs_result, belongs_expected)


class TestCholeskyMetric(TestCase):
    cls = CholeskyMetric

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

    testing_data = TestDataCholeskyMetric()

    def test_diag_inner_product(
        self, n, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        self.assertAllClose(
            self.cls(n).diag_inner_product(
                gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
            ),
            gs.array(expected),
        )

    def test_strictly_lower_inner_product(
        self, n, tangent_vec_a, tangent_vec_b, expected
    ):
        self.assertAllClose(
            self.cls(n).strictly_lower_inner_product(
                gs.array(tangent_vec_a), gs.array(tangent_vec_b)
            ),
            gs.array(expected),
        )

    def test_diag_inner_product(
        self, n, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        self.assertAllClose(
            self.cls(n).inner_product(
                gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
            ),
            gs.array(expected),
        )

    def test_exp(self, n, tangent_vec, base_point, expected):
        self.assertAllClose(
            self.cls(n).exp(
                gs.array(tangent_vec), gs.array(base_point), gs.array(expected)
            )
        )

    def test_log(self, n, point, base_point, expected):
        self.assertAllClose(
            self.cls(n).log(gs.array(point), gs.array(base_point)), gs.array(expected)
        )

    def test_squared_dist(self, n, point_a, point_b, expected):
        self.assertAllClose(
            self.cls(n).squared_dist(gs.array(point_a), gs.array(point_b)),
            gs.array(expected),
        )

    def test_exp_vectorization(self):
        """Test exp map vectorization"""
        L = self.space.random_point(5)
        X = self.space.ambient_space.random_point(5)
        D_L = Matrices.to_diagonal(L)
        D_X = Matrices.to_diagonal(X)
        inv_D_L = gs.linalg.inv(D_L)
        exp_expected = (
            Matrices.to_strictly_lower_triangular(L)
            + Matrices.to_strictly_lower_triangular(X)
            + gs.matmul(D_L, SPDMatrices(2).expm(gs.matmul(D_X, inv_D_L)))
        )
        exp_result = self.metric_cholesky.exp(X, L)
        belongs_result = gs.all(self.space.belongs(exp_result))
        belongs_expected = True

        self.assertAllClose(exp_expected, exp_result)
        self.assertAllClose(belongs_expected, belongs_result)

    def test_log_vectorization(self):
        """Test log map"""
        K = self.space.random_point(5)
        L = self.space.random_point(5)
        D_K = Matrices.to_diagonal(K)
        D_L = Matrices.to_diagonal(L)
        inv_D_L = gs.linalg.inv(D_L)
        log_result = self.metric_cholesky.log(K, L)
        log_expected = (
            Matrices.to_strictly_lower_triangular(K)
            - Matrices.to_strictly_lower_triangular(L)
            + gs.matmul(D_L, SPDMatrices(2).logm(gs.matmul(inv_D_L, D_K)))
        )
        belongs_result = gs.all(self.space.ambient_space.belongs(log_result))
        belongs_expected = True
        self.assertAllClose(log_expected, log_result)
        self.assertAllClose(belongs_expected, belongs_result)
