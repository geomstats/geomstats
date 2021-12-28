"""Unit tests for the vector space of symmetric matrices."""

import math
import random
import warnings

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.conftest import Parametrizer, TestData


class TestSymmetricMatrices(geomstats.tests.TestCase, metaclass=Parametrizer):
    """Test of SymmetricMatrices methods."""

    class TestingData(TestData):
        """Data class for Testing Symmetric Matrices"""

        def belongs_data(self):
            smoke_data = [
                dict(n=2, mat=[[1.0, 2.0], [2.0, 1.0]], expected=True),
                dict(n=2, mat=[[1.0, 1.0], [2.0, 1.0]], expected=False),
                dict(
                    n=3,
                    mat=[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                    expected=True,
                ),
                dict(
                    n=2,
                    mat=[[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]],
                    expected=[True, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def basis_data(self):
            smoke_data = [
                dict(n=1, basis=[[[1.0]]]),
                dict(
                    n=2,
                    basis=[
                        [[1.0, 0.0], [0, 0]],
                        [[0, 1.0], [1.0, 0]],
                        [[0, 0.0], [0, 1.0]],
                    ],
                ),
            ]
            return self.generate_tests(smoke_data)

        def expm_data(self):
            smoke_data = [
                dict(mat=[[0.0, 0.0], [0.0, 0.0]], expected=[[1.0, 0.0], [0.0, 1.0]])
            ]
            return self.generate_tests(smoke_data)

        def powerm_data(self):
            smoke_data = [
                dict(
                    mat=[[1.0, 2.0], [2.0, 3.0]],
                    power=1.0,
                    expected=[[1.0, 2.0], [2.0, 3.0]],
                ),
                dict(
                    mat=[[1.0, 2.0], [2.0, 3.0]],
                    power=2.0,
                    expected=[[5.0, 8.0], [8.0, 13.0]],
                ),
            ]
            return self.generate_tests(smoke_data)

        def projection_data(self):
            smoke_data = [
                dict(n=1, num_points=1),
                dict(n=2, num_points=1),
                dict(n=1, num_points=10),
                dict(n=10, num_points=10),
            ]
            return self.generate_tests(smoke_data)

        def dim_data(self):

            smoke_data = [dict(n=1, dim=1), dict(n=2, dim=3), dict(n=5, dim=15)]

            random_n = random.sample(range(1, 1000), 500)
            rt_data = [(n, (n * (n + 1)) // 2) for n in random_n]
            return self.generate_tests(smoke_data, rt_data)

        def to_vector_data(self):
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
                    vec=[
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    ],
                ),
            ]
            return self.generate_tests(smoke_data)

        def from_vector_data(self):
            smoke_data = [
                dict(n=1, vec=[1.0], mat=[[1.0]]),
                dict(
                    n=3,
                    vec=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    mat=[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                ),
                dict(
                    n=3,
                    vec=[
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    ],
                    mat=[
                        [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
                        [[7.0, 8.0, 9.0], [8.0, 10.0, 11.0], [9.0, 11.0, 12.0]],
                    ],
                ),
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestingData()

    def test_belongs(self, n, mat, expected):
        result = SymmetricMatrices(n).belongs(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_basis(self, n, basis):
        self.assertAllClose(SymmetricMatrices(n).get_basis(), gs.array(basis))

    def test_expm(self, mat, expected):
        result = SymmetricMatrices.expm(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_powerm(self, mat, power, expected):
        result = SymmetricMatrices.powerm(gs.array(mat), power)
        self.assertAllClose(result, gs.array(expected))

    def test_projection(self, n, num_points):
        space = SymmetricMatrices(n)
        shape = (num_points, n, n)
        result = gs.all(helper.test_projection_and_belongs(space, shape))
        self.assertTrue(result)

    def test_from_vector(self, n, vec, expected):
        result = SymmetricMatrices(n).from_vector(gs.array(vec))
        self.assertAllClose(result, gs.array(expected))

    def test_to_vector(self, n, mat, expected):
        result = SymmetricMatrices(n).to_vector(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_dim(self, n, expected_dim):
        self.assertAllClose(SymmetricMatrices(n).dim, expected_dim)
