"""Unit tests for the manifold of lower triangular matrices with positive diagonal elmeents"""

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


class TestPositiveLowerTriangularMatrices(geomstats.tests.TestCase):
    """Test of Cholesky methods."""

    def setUp(self):
        """Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

        gs.random.seed(1234)

        self.n = 3
        self.space = PositiveLowerTriangularMatrices(n=self.n)
        self.metric_cholesky = CholeskyMetric(n=self.n)
        self.n_samples = 5

    def test_belongs(self):
        """Test of belongs method."""
        mats = gs.array([[1.0, 0.0], [-1.0, 3.0]])
        result = PositiveLowerTriangularMatrices(2).belongs(mats)
        expected = True
        self.assertAllClose(result, expected)

        mats = gs.array([[1.0, -1.0], [-1.0, 3.0]])
        result = PositiveLowerTriangularMatrices(2).belongs(mats)
        expected = False
        self.assertAllClose(result, expected)

        mats = gs.array([[-1.0, 0.0], [-1.0, 3.0]])
        result = PositiveLowerTriangularMatrices(2).belongs(mats)
        expected = False
        self.assertAllClose(result, expected)

        mats = gs.eye(3)
        result = PositiveLowerTriangularMatrices(2).belongs(mats)
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

        result = PositiveLowerTriangularMatrices(2).belongs(mats_2dim)
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
        point = self.space.random_point()
        gram = self.space.gram(point)

        gram_result = gram
        gram_expected = gs.matmul(point, Matrices.transpose(point))
        self.assertAllClose(gram_expected, gram_result)

        belongs_result = SPDMatrices(self.n).belongs(gram)
        belongs_expected = True
        self.assertAllClose(belongs_expected, belongs_result)

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
        """Test differential of gram"""
        base_point = self.space.random_point()
        tangent_vec = self.space.ambient_space.random_point()

        diff1 = gs.matmul(base_point, Matrices.transpose(tangent_vec))
        diff2 = gs.matmul(tangent_vec, Matrices.transpose(base_point))
        diff = diff1 + diff2

        belongs_result = SymmetricMatrices(self.n).belongs(diff)
        belongs_expected = True
        self.assertAllClose(belongs_expected, belongs_result)

        diff_expected = diff
        diff_result = self.space.differential_gram(tangent_vec, base_point)
        self.assertAllClose(diff_expected, diff_result)

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
        pass

    
    def test_diag_inner_product(self):
        """Test inner product on diag part"""
        pass

    def test_strictly_lower_inner_product(self):
        """Test inner product on diag part"""
        pass

    def test_inner_product(self):
        """Test inner product"""
        pass

    def test_exp_metric(self):
        """Test exp map"""
        pass
    def test_log_metric(self):
        """Test log map"""
        pass

    def test_squared_dist(self):
        """Test squared dist function"""
        pass