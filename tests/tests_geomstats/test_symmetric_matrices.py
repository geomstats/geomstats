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


def belongs_data():
    smoke_data = [
        dict(n=2, mat=[[1.0, 2.0], [2.0, 1.0]], expected=[True]),
        dict(n=2, mat=[[1.0, 1.0], [2.0, 1.0]], expected=[False]),
        dict(
            n=3,
            mat=[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
            expected=[True],
        ),
        dict(
            n=2,
            mat=[[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]],
            expected=[True, False],
        ),
    ]
    return generate_tests(smoke_data)


def basis_data():
    smoke_data = [
        dict(n=1, basis=[[[1.0]]]),
        dict(
            n=2,
            basis=[[[1.0, 0.0], [0, 0]], [[0, 1.0], [1.0, 0]], [[0, 0.0], [0, 1.0]]],
        ),
    ]
    return generate_tests(smoke_data)


def expm_data():
    smoke_data = [dict(mat=[[0.0, 0.0], [0.0, 0.0]], expected=[[1.0, 0.0], [0.0, 1.0]])]
    return generate_tests(smoke_data)


def powerm_data():
    smoke_data = [
        dict(
            mat=[[1.0, 2.0], [2.0, 3.0]], power=1.0, expected=[[1.0, 2.0], [2.0, 3.0]]
        ),
        dict(
            mat=[[1.0, 2.0], [2.0, 3.0]], power=2.0, expected=[[5.0, 8.0], [8.0, 13.0]]
        ),
    ]
    return generate_tests(smoke_data)


def dim_data():
    random_n = random.sample(range(1, 1000), 500)
    smoke_data = [dict(n=1, dim=1), dict(n=2, dim=3), dict(n=5, dim=15)]
    rt_data = [(n, (n * (n + 1)) // 2) for n in random_n]
    return generate_tests(smoke_data, rt_data)


def to_vector_data():
    smoke_data = [
        dict(n=1, mat=[[1.0]], vec=[1.0]),
        dict(
            n=3,
            mat=[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
            vec=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        dict(
            n=3,
            mat=[
                [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [8.0, 10.0, 11.0], [9.0, 11.0, 12.0]],
            ],
            vec=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
        ),
    ]
    return generate_tests(smoke_data)


def from_vector_data():
    smoke_data = [
        dict(n=1, vec=[1.0], mat=[[1.0]]),
        dict(
            n=3,
            vec=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            mat=[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
        ),
        dict(
            n=3,
            vec=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
            mat=[
                [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [8.0, 10.0, 11.0], [9.0, 11.0, 12.0]],
            ],
        ),
    ]
    return generate_tests(smoke_data)


def projection_data():
    smoke_data = [
        dict(n=1, num_points=1),
        dict(n=2, num_points=1),
        dict(n=1, num_points=10),
        dict(n=10, num_points=10),
    ]
    return generate_tests(smoke_data)


class TestSymmetricMatrices(geomstats.tests.TestCase):
    """Test of SymmetricMatrices methods."""

    @pytest.mark.parametrize("n, mat, expected", belongs_data())
    def test_belongs(self, n, mat, expected):
        """Test of belongs method."""
        result = SymmetricMatrices(n).belongs(gs.array(mat))
        self.assertAllClose(result, expected)

    @pytest.mark.parametrize("n, basis", basis_data())
    def test_basis(self, n, basis):
        """Test of belongs method."""
        self.assertAllClose(SymmetricMatrices(n).get_basis(), gs.array(basis))

    @pytest.mark.parametrize("mat, expected", expm_data())
    def test_expm(self, mat, expected):
        """Test of expm method."""
        print("mat", mat)
        print("expected", expected)
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
