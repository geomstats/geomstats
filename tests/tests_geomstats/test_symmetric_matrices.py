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


class TestSymmetricMatrices(geomstats.tests.TestCase):
    """Test of SymmetricMatrices methods."""

    def setup_method(self):
        """Set up the test."""
        warnings.simplefilter("ignore", category=ImportWarning)

        gs.random.seed(1234)

        self.n = 3
        self.space = SymmetricMatrices(self.n)

    def test_belongs(self):
        """Test of belongs method."""
        sym_n = self.space
        mat_sym = gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        mat_not_sym = gs.array([[1.0, 0.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        result = sym_n.belongs(mat_sym)
        expected = True
        self.assertAllClose(result, expected)

        result = sym_n.belongs(mat_not_sym)
        expected = False
        self.assertAllClose(result, expected)

    def test_basis(self):
        """Test of belongs method."""
        sym_n = SymmetricMatrices(2)
        mat_sym_1 = gs.array([[1.0, 0.0], [0, 0]])
        mat_sym_2 = gs.array([[0, 1.0], [1.0, 0]])
        mat_sym_3 = gs.array([[0, 0.0], [0, 1.0]])
        expected = gs.stack([mat_sym_1, mat_sym_2, mat_sym_3])
        result = sym_n.basis
        self.assertAllClose(result, expected)

    def test_expm(self):
        """Test of expm method."""
        sym_n = SymmetricMatrices(self.n)
        v = gs.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = sym_n.expm(v)
        c = math.cosh(1)
        s = math.sinh(1)
        e = math.exp(1)
        expected = gs.array([[c, s, 0.0], [s, c, 0.0], [0.0, 0.0, e]])

        four_dim_v = gs.broadcast_to(v, (2, 2) + v.shape)
        four_dim_expected = gs.broadcast_to(expected, (2, 2) + expected.shape)
        four_dim_result = sym_n.expm(four_dim_v)

        self.assertAllClose(result, expected)
        self.assertAllClose(four_dim_result, four_dim_expected)

    def test_powerm(self):
        """Test of powerm method."""
        sym_n = SymmetricMatrices(self.n)
        expected = gs.array(
            [[[1, 1.0 / 4.0, 0.0], [1.0 / 4, 2.0, 0.0], [0.0, 0.0, 1.0]]]
        )

        power = gs.array(1.0 / 2.0)

        result = sym_n.powerm(expected, power)
        result = gs.matmul(result, gs.transpose(result, (0, 2, 1)))
        self.assertAllClose(result, expected)

    def test_from_vector(self, n, vec, expected):
        """Test from vector"""
        result = SymmetricMatrices(n).from_vector(gs.array(vec))
        self.assertAllClose(result, gs.array(expected))

    @pytest.mark.parametrize(
        "n,  num_points",
        [
            (1, 1),
            (2, 10),
            (10, 100),
        ],
    )
    def test_to_vector_and_from_vector(self, n, num_points):
        """Testing to_vector and from_vector in chained way"""
        space = SymmetricMatrices(n)
        vector = gs.random.rand(num_points, space.dim)
        result = space.to_vector(space.from_vector(vector))
        self.assertTrue(gs.allclose(result, vector))

        mat = space.random_point(num_points)
        result = space.from_vector(space.to_vector(mat))
        self.assertTrue(gs.allclose(result, mat))

    def test_projection_and_belongs(self):
        shape = (2, self.n, self.n)
        result = helper.test_projection_and_belongs(self.space, shape)
        for res in result:
            self.assertTrue(res)

    @pytest.mark.parametrize("n, vec, expected", from_vector_data())
    def test_from_vector(self, n, vec, expected):
        """Test from vector"""
        result = SymmetricMatrices(n).from_vector(gs.array(vec))
        self.assertAllClose(result, gs.array(expected))

    @pytest.mark.parametrize("n, mat, expected", to_vector_data())
    def test_to_vector(self, n, mat, expected):
        """Test to vector."""
        result = SymmetricMatrices(n).to_vector(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    @pytest.mark.parametrize("n, expected_dim", dim_data())
    def test_dim(self, n, expected_dim):
        self.assertAllClose(SymmetricMatrices(n).dim, expected_dim)
