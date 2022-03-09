"""Unit tests for the vector space of symmetric matrices."""

import random

import geomstats.backend as gs
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.conftest import TestCase
from tests.data_generation import _VectorSpaceTestData
from tests.parametrizers import VectorSpaceParametrizer


class TestSymmetricMatrices(TestCase, metaclass=VectorSpaceParametrizer):
    """Test of SymmetricMatrices methods."""

    space = SymmetricMatrices

    class SymmetricMatricesTestData(_VectorSpaceTestData):
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

        def basis_belongs_data(self):
            space_args_list = [(n,) for n in random.sample(range(1, 10), 5)]
            return self._basis_belongs_data(space_args_list)

        def basis_cardinality_data(self):
            space_args_list = [(n,) for n in random.sample(range(1, 10), 5)]
            return self._basis_cardinality_data(space_args_list)

        def projection_belongs_data(self):
            space_args_list = [(n,) for n in random.sample(range(1, 10), 5)]
            n_samples_list = random.sample(range(1, 10), 5)
            shapes = [(n, n) for (n,), in zip(space_args_list)]
            return self._projection_belongs_data(
                space_args_list, shapes, n_samples_list
            )

        def to_tangent_is_tangent_data(self):
            space_args_list = [(n,) for n in random.sample(range(1, 10), 5)]
            tangent_shapes_list = [(n, n) for (n,) in space_args_list]
            n_vecs_list = random.sample(range(1, 10), 5)
            return self._to_tangent_is_tangent_data(
                SymmetricMatrices, space_args_list, tangent_shapes_list, n_vecs_list
            )

        def random_point_belongs_data(self):
            smoke_space_args_list = [(1,), (2,), (3,)]
            smoke_n_points_list = [1, 1, 10]
            space_args_list = [(n,) for n in random.sample(range(1, 10), 5)]
            n_points_list = random.sample(range(1, 100), 5)

            return self._random_point_belongs_data(
                smoke_space_args_list,
                smoke_n_points_list,
                space_args_list,
                n_points_list,
                gs.atol * 1000,
            )

    testing_data = SymmetricMatricesTestData()

    def test_belongs(self, n, mat, expected):
        result = SymmetricMatrices(n).belongs(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_basis(self, n, basis):
        self.assertAllClose(SymmetricMatrices(n).basis, gs.array(basis))

    def test_expm(self, mat, expected):
        result = SymmetricMatrices.expm(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_powerm(self, mat, power, expected):
        result = SymmetricMatrices.powerm(gs.array(mat), power)
        self.assertAllClose(result, gs.array(expected))

    def test_from_vector(self, n, vec, expected):
        result = SymmetricMatrices(n).from_vector(gs.array(vec))
        self.assertAllClose(result, gs.array(expected))

    def test_to_vector(self, n, mat, expected):
        result = SymmetricMatrices(n).to_vector(gs.array(mat))
        self.assertAllClose(result, gs.array(expected))

    def test_dim(self, n, expected_dim):
        self.assertAllClose(SymmetricMatrices(n).dim, expected_dim)
