"""Unit tests for the vector space of symmetric matrices."""

import math
import random
import warnings

import pytest

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.conftest import generate_tests


def dim_data():
    random_n = random.sample(range(1, 1000), 500)
    smoke_data = [(1, 1), (2, 3), (5, 15)]
    rt_data = [(n, (n * (n + 1)) // 2) for n in random_n]
    return generate_tests(smoke_data, rt_data)


def to_vector_data():
    smoke_data = [
        (1, [[1.0]], [1.0]),
        (
            3,
            [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        (
            3,
            [
                [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [8.0, 10.0, 11.0], [9.0, 11.0, 12.0]],
            ],
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
        ),
    ]
    return generate_tests(smoke_data)


def from_vector_data():
    smoke_data = [
        (1, [1.0], [[1.0]]),
        (
            3,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
        ),
        (
            3,
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
            [
                [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [8.0, 10.0, 11.0], [9.0, 11.0, 12.0]],
            ],
        ),
    ]
    return generate_tests(smoke_data)


def projection_data():
    smoke_data = [(1, 1), (2, 1), (1, 10), (10, 10)]
    return generate_tests(smoke_data)


def powerm_data():
    smoke_data = [
        ([[1.0, 2.0], [2.0, 3.0]], 1.0, [[1.0, 2.0], [2.0, 3.0]]),
        ([[1.0, 2.0], [2.0, 3.0]], 2.0, [[5.0, 8.0], [8.0, 13.0]]),
    ]
    return generate_tests(smoke_data)


class TestSymmetricMatrices(geomstats.tests.TestCase):
    """Test of SymmetricMatrices methods."""

    @pytest.mark.parametrize("n, mat, expected", belongs_data())
    def test_belongs(self, n, mat, expected):
        """Test of belongs method."""
        result = SymmetricMatrices(n).belongs(gs.array(mat))
        self.assertAllClose(result, expected)

    @pytest.mark.parametrize("n, expected", basis_data())
    def test_basis(self, n, expected):
        """Test of belongs method."""
        self.assertAllClose(SymmetricMatrices(n), gs.array(expected))

    @pytest.mark.parametrize("mat, expected", expm_data())
    def test_expm(self, mat, expected):
        """Test of expm method."""
        result = SymmetricMatrices.expm(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    @pytest.mark.parametrize("mat, power, expected", powerm_data())
    def test_powerm(self, mat, power, expected):
        """Test powerm method."""
        result = SymmetricMatrices.powerm(gs.array(mat), power)
        self.assertAllClose(result, gs.array(expected))

    @pytest.mark.parametrize("n, num_points", projection_data())
    def test_projection(self, n, num_points):
        "Test projection"
        space = SymmetricMatrices(n)
        shape = (num_points, n, n)
        result = gs.all(helper.test_projection_and_belongs(space, shape))
        self.assertTrue(result)

    @pytest.mark.parametrize("n, vec, expected", from_vector_data())
    def test_from_vector(self, n, vec, expected):
        """Test from vector."""
        result = SymmetricMatrices(n).from_vector(gs.array(vec))
        self.assertAllClose(result, gs.array(expected))

    @pytest.mark.parametrize("n, mat, expected", to_vector_data())
    def test_to_vector(self, n, mat, expected):
        """Test to vector."""
        result = SymmetricMatrices(n).to_vector(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    @pytest.mark.parametrize("n, expected_dim", dim_data())
    def test_dim(self, n, expected_dim):
        """Test dim."""
        self.assertAllClose(SymmetricMatrices(n).dim, expected_dim)
