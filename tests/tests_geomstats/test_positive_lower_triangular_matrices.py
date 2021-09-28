"""Unit tests for the manifold of lower triangular matrices with positive diagonal elmeents"""

from geomstats.geometry.poincare_ball import SQRT_2
from geomstats.geometry import matrices
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.geometry.matrices import Matrices
import math
import warnings

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatrices,
    CholeskyMetric,
)
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices

EULER = gs.exp(1.0)


class TestPositiveLowerTriangularMatrices(geomstats.tests.TestCase):
    """Test of Cholesky methods."""

    def setUp(self):
        """Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

        gs.random.seed(1234)

        self.n = 2
        self.space = PositiveLowerTriangularMatrices(n=self.n)
        self.metric_cholesky = CholeskyMetric(n=self.n)
        self.n_samples = 5

    def test_belongs(self):
        """Test of belongs method."""
        mats = gs.array([[1.0, 0.0], [-1.0, 3.0]])
        result = self.space.belongs(mats)
        expected = True
        self.assertAllClose(result, expected)

        mats = gs.array([[1.0, -1.0], [-1.0, 3.0]])
        result = self.space.belongs(mats)
        expected = False
        self.assertAllClose(result, expected)

        mats = gs.array([[-1.0, 0.0], [-1.0, 3.0]])
        result = self.space.belongs(mats)
        expected = False
        self.assertAllClose(result, expected)

        mats = gs.eye(3)
        result = self.space.belongs(mats)
        expected = False
        self.assertAllClose(result, expected)

    def test_belongs_vectorization(self):
        """Test of belongs method."""
        mats_2dim = gs.array(
            [
                [[1.0, 0], [0, 1.0]],
                [[1.0, 2.0], [2.0, 1.0]],
                [[-1.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ]
        )

        mats_3dim = gs.array(
            [
                [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[-1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )

        result = self.space.belongs(mats_2dim)
        expected = gs.array([True, False, False, False])
        self.assertAllClose(result, expected)

        result = PositiveLowerTriangularMatrices(3).belongs(mats_3dim)
        expected = gs.array([False, False, True, False])
        self.assertAllClose(result, expected)

    def test_random_point_and_belongs(self):
        """Test of random_point and belongs methods."""
        point = self.space.random_point()
        result = self.space.belongs(point)
        expected = True
        self.assertAllClose(result, expected)

    def test_random_point_and_belongs_vectorization(self):
        """Test of random_point and belongs methods."""
        points = self.space.random_point(4)
        result = self.space.belongs(points)
        expected = gs.array([True] * 4)
        self.assertAllClose(result, expected)

    def test_gram(self):
        """Test gram method for single point"""
        L = gs.array([[1.0, 0.0], [2.0, 1.0]])
        gram_expected = gs.array([[1.0, 2.0], [2.0, 5.0]])
        gram_result = self.space.gram(L)
        self.assertAllClose(gram_expected, gram_result)

    def test_gram_vectorization(self):
        """Test gram method for batch of points"""
        point = self.space.random_point(n_samples=5)
        gram = self.space.gram(point)

        gram_result = gram
        gram_expected = gs.matmul(point, Matrices.transpose(point))
        self.assertAllClose(gram_expected, gram_result)

        belongs_result = gs.all(SPDMatrices(self.n).belongs(gram))
        belongs_expected = True
        self.assertAllClose(belongs_expected, belongs_result)

    def test_differential_gram(self):
        """Test differential of gram for single point"""
        L = gs.array([[1.0, 0.0], [2.0, 1.0]])
        X = gs.array([[-1.0, 0.0], [2.0, -1.0]])
        diff_gram_result = self.space.differential_gram(X, L)
        diff_gram_expected = gs.array([[-2.0, 0.0], [0.0, 6.0]])
        self.assertAllClose(diff_gram_expected, diff_gram_result)

    def test_differential_gram_vectorization(self):
        """Test differential of gram"""
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

    def test_inv_differential_gram(self):
        """Test inverse differential of gram"""
        L = gs.array([[1.0, 0.0], [2.0, 2.0]])
        W = gs.array([[1.0, 2.0], [2.0, 5.0]])
        inv_diff_gram_expected = gs.array([[0.0, 0.0], [1.0, 0.25]])
        inv_diff_gram_result = self.space.inverse_differential_gram(W, L)
        self.assertAllClose(inv_diff_gram_expected, inv_diff_gram_result)

    def test_inv_differential_gram_belongs(self):
        """Test inverse differential of gram belongs """
        L = self.space.random_point(5)
        W = SymmetricMatrices(2).random_point(5)
        inv_diff_gram_result = self.space.inverse_differential_gram(W, L)
        belongs_result = gs.all(self.space.ambient_space.belongs(inv_diff_gram_result))
        belongs_expected = True
        self.assertAllClose(belongs_result, belongs_expected)
     
    def test_diag_inner_product(self):
        """Test inner product on diag part"""
        X = gs.array([[1.0, 0.0], [-2.0, -1.0]])
        Y = gs.array([[2.0, 0.0], [-3.0, -1.0]])
        L = gs.array([[SQRT_2, 0.0], [-3.0, 1.0]])
        dip_result = self.metric_cholesky.diag_inner_product(X, Y, L)
        dip_expected = 2
        self.assertAllClose(dip_expected, dip_result)

    def test_strictly_lower_inner_product(self):
        """Test inner product on diag part"""
        X = gs.array([[1.0, 0.0], [-2.0, -1.0]])
        Y = gs.array([[2.0, 0.0], [-3.0, -1.0]])
        dip_result = self.metric_cholesky.strictly_lower_inner_product(X, Y)
        dip_expected = 6
        self.assertAllClose(dip_expected, dip_result)

    def test_inner_product(self):
        """Test inner product"""
        X = gs.array([[1.0, 0.0], [-2.0, -1.0]])
        Y = gs.array([[2.0, 0.0], [-3.0, -1.0]])
        L = gs.array([[SQRT_2, 0.0], [-3.0, 1.0]])
        ip_result = self.metric_cholesky.inner_product(X, Y, L)
        ip_expected = 2 + 6
        self.assertAllClose(ip_expected, ip_result)

    def test_inner_product_vectorization(self):
        """Test inner product"""
        n_samples = 5
        X = self.space.ambient_space.random_point(n_samples)
        Y = self.space.ambient_space.random_point(n_samples)
        L = self.space.random_point(n_samples)
        inv_D_L = gs.linalg.inv(Matrices.to_diagonal(L))
        D_X = Matrices.to_diagonal(X)
        D_Y = Matrices.to_diagonal(Y)
        ip_result = self.metric_cholesky.inner_product(X, Y, L)
        ip_expected = (
            Matrices.frobenius_product(
                Matrices.to_strictly_lower_triangular(X),
                Matrices.to_strictly_lower_triangular(Y),
            )
            + Matrices.frobenius_product(
                gs.matmul(inv_D_L, D_X), gs.matmul(inv_D_L, D_Y)
            )
        )
        ip_shape_result = ip_result.shape[0]
        ip_shape_expected = n_samples

        self.assertAllClose(ip_expected, ip_result)
        self.assertAllClose(ip_shape_expected, ip_shape_result)

    def test_exp(self):
        """Test exp map"""
        L = gs.array([[1.0, 0.0], [2.0, 3.0]])
        X = gs.array([[-1.0, 0.0], [2.0, 3.0]])
        exp_expected = gs.array([[1 / EULER, 0.0], [4.0, 2 * gs.exp(1.5)]])
        exp_result = self.metric_cholesky.exp(X, L)
        self.assertAllClose(exp_expected, exp_result)

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

    def test_log(self):
        """Test log map"""
        K = gs.array([[EULER, 0.0], [2.0, EULER ** 3]])
        L = gs.array([[EULER ** 3, 0.0], [4.0, EULER ** 4]])
        log_result = self.metric_cholesky.log(K, L)
        log_expected = gs.array([[0.0, 0.0], [-2.0, 0.0]]) + gs.array(
            [[-2.0 * EULER ** 3, 0.0], [0.0, -1 * EULER ** 4]]
        )
        self.assertAllClose(log_expected, log_result)

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

    def test_squared_dist(self):
        """Test squared dist function"""
        K = gs.array([[EULER, 0.0], [2.0, EULER ** 3]])
        L = gs.array([[EULER ** 3, 0.0], [4.0, EULER ** 4]])

        squared_dist_result = self.metric_cholesky.squared_dist(K, L)
        squared_dist_expected = 4 + 5

        batch_K = gs.array(
            [[[EULER, 0.0], [2.0, EULER ** 3]], [[EULER, 0.0], [4.0, EULER ** 3]]]
        )
        batch_L = gs.array(
            [
                [[EULER ** 3, 0.0], [4.0, EULER ** 4]],
                [[EULER ** 3, 0.0], [7.0, EULER ** 4]],
            ]
        )
        batch_squared_dist_result = self.metric_cholesky.squared_dist(batch_K, batch_L)
        batch_squared_dist_expected = gs.array([4 + 5, 9 + 5])
        self.assertAllClose(squared_dist_expected, squared_dist_result)
        self.assertAllClose(batch_squared_dist_expected, batch_squared_dist_result)
