"""Unit tests for the manifold of lower triangular matrices with positive diagonal elmeents"""

import math
import warnings

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.cholesky import CholeskySpace, CholeskyMetric


class TestPositiveLowerTriangularMatrices(geomstats.tests.TestCase):
    """Test of Cholesky methods."""

    def setUp(self):
        """Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

        gs.random.seed(1234)

        self.n = 3
        self.space = CholeskySpace(n=self.n)
        self.metric_cholesky = CholeskyMetric(n=self.n)
        self.n_samples = 4

    def test_belongs(self):
        """Test of belongs method."""
        mats = gs.array([[1.0, 0.0], [-1.0, 3.0]])
        result = CholeskySpace(2).belongs(mats)
        expected = True
        self.assertAllClose(result, expected)

        mats = gs.array([[1.0, -1.0], [-1.0, 3.0]])
        result = CholeskySpace(2).belongs(mats)
        expected = False
        self.assertAllClose(result, expected)

        mats = gs.array([[-1.0, 0.0], [-1.0, 3.0]])
        result = CholeskySpace(2).belongs(mats)
        expected = False
        self.assertAllClose(result, expected)

        mats = gs.eye(3)
        result = CholeskySpace(2).belongs(mats)
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

        result = CholeskySpace(2).belongs(mats_2dim)
        expected = gs.array([True, False, False, False])
        self.assertAllClose(result, expected)

        result = CholeskySpace(3).belongs(mats_3dim)
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

    def test_to_vec_and_from_vec(self):
        """Test for matrix to vector and vector to matrix conversions."""
        chol_mat_1 = gs.array([[1.0, 0.0, 0.0], [0.6, 7.0, 0.0], [-3.0, 0.0, 8.0]])
        vector_1 = self.space.to_vector(chol_mat_1)
        result_1 = self.space.from_vector(vector_1)
        expected_1 = chol_mat_1
        self.assertTrue(gs.allclose(result_1, expected_1))

        chol_mat_1 = gs.array([[1.0, 0.0, 1.0], [0.6, 7.0, 0.0], [-3.0, 0.0, 8.0]])
        vector_1 = self.space.to_vector(chol_mat_1)
        result_1 = self.space.from_vector(vector_1)
        expected_1 = chol_mat_1
        self.assertTrue(gs.allclose(result_1, expected_1))

        vector_2 = gs.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        chol_mat_2 = self.space.from_vector(vector_2)
        result_2 = self.space.to_vector(chol_mat_2)
        expected_2 = vector_2

        self.assertTrue(gs.allclose(result_2, expected_2))

    def test_to_vec_and_from_vec_vectorization(self):
        """Test of vectorization."""
        n_samples = self.n_samples
        vector = gs.exp(gs.random.rand(n_samples, 6))
        chol_mat = self.space.from_vector(vector)
        result = self.space.to_vector(chol_mat)
        expected = vector

        self.assertTrue(gs.allclose(result, expected))

        chol_mat = self.space.random_point(n_samples)
        vector = self.space.to_vector(chol_mat)
        result = self.space.from_vector(vector)
        expected = chol_mat

        self.assertTrue(gs.allclose(result, expected))
