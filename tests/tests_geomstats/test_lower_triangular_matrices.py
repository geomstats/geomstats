"""Unit tests for the vector space of lower triangular matrices."""

import math
import warnings

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices


class TestLowerTriangularMatrices(geomstats.tests.TestCase):
    """Test of LowerTriangularMatrices methods."""

    def setup_method(self):
        """Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

        gs.random.seed(1234)

        self.n = 3
        self.space = LowerTriangularMatrices(self.n)

    def test_belongs(self):
        """Test of belongs method."""
        mats = gs.array([[1.0, 0.0], [-1.0, 3.0]])
        result = LowerTriangularMatrices(2).belongs(mats)
        expected = True
        self.assertAllClose(result, expected)

        mats = gs.array([[1.0, -1.0], [-1.0, 3.0]])
        result = LowerTriangularMatrices(2).belongs(mats)
        expected = False
        self.assertAllClose(result, expected)

        mats = gs.array([[-1.0, 0.0], [0.0, -3.0]])
        result = LowerTriangularMatrices(2).belongs(mats)
        expected = True
        self.assertAllClose(result, expected)

        mats = gs.eye(3)
        result = LowerTriangularMatrices(2).belongs(mats)
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

        result = LowerTriangularMatrices(2).belongs(mats_2dim)
        expected = gs.array([True, False, True, True])
        self.assertAllClose(result, expected)

        result = LowerTriangularMatrices(3).belongs(mats_3dim)
        expected = gs.array([False, True, True, True])
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

    def test_to_vector(self):
        """Test for matrix to vector"""
        chol_mat = gs.array([[1.0, 0.0, 0.0], [0.6, 7.0, 0.0], [-3.0, 0.0, 8.0]])
        result = self.space.to_vector(chol_mat)
        expected = gs.array([1.0, 0.6, 7.0, -3.0, 0.0, 8.0])
        self.assertTrue(gs.allclose(result, expected))

    def test_to_vector_vectorization(self):
        """Test of to vector function with vectorization."""
        chol_mat = gs.array(
            [
                [[1.0, 0.0, 0.0], [0.6, 7.0, 0.0], [-3.0, 0.0, 8.0]],
                [[2.0, 0.0, 0.0], [2.6, 7.0, 0.0], [-3.0, 0.0, 28.0]],
            ]
        )
        result = self.space.to_vector(chol_mat)
        expected = gs.array(
            [[1.0, 0.6, 7.0, -3.0, 0.0, 8.0], [2.0, 2.6, 7.0, -3.0, 0.0, 28.0]]
        )
        self.assertTrue(gs.allclose(result, expected))

    def test_get_basis(self):
        """Test of get basis function"""
        space2 = LowerTriangularMatrices(2)
        result = space2.get_basis()
        expected = gs.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )
        self.assertAllClose(result, expected)

        space10 = LowerTriangularMatrices(10)
        result = space10.get_basis()
        self.assertAllClose(gs.shape(result), (55, 10, 10))

    def test_projection(self):
        """Test of projection function"""
        point1 = gs.array([[2.0, 1.0], [1.0, 2.0]])
        point2 = gs.array([[1.0, 0.0], [0.0, 1.0]])

        space2 = LowerTriangularMatrices(2)
        result = space2.projection(point1)
        expected = gs.array([[2.0, 0.0], [1.0, 2.0]])
        self.assertAllClose(result, expected)

        result = space2.projection(point2)
        expected = point2
        self.assertAllClose(result, expected)
