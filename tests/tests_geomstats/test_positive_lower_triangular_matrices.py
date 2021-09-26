"""Unit tests for the manifold of lower triangular matrices with positive diagonal elmeents"""

import math
import warnings

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatrices,
    CholeskyMetric,
)


class TestPositiveLowerTriangularMatrices(geomstats.tests.TestCase):
    """Test of Cholesky methods."""

    def setUp(self):
        """Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

        gs.random.seed(1234)

        self.n = 3
        self.space = PositiveLowerTriangularMatrices(n=self.n)
        self.metric_cholesky = CholeskyMetric(n=self.n)
        self.n_samples = 4

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
