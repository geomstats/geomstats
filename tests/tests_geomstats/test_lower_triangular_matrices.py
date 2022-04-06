"""Unit tests for the vector space of lower triangular matrices."""
import random

import geomstats.backend as gs
from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from tests.conftest import Parametrizer
from tests.data_generation import _VectorSpaceTestData
from tests.geometry_test_cases import VectorSpaceTestCase


class TestLowerTriangularMatrices(VectorSpaceTestCase, metaclass=Parametrizer):
    """Test of LowerTriangularMatrices methods."""

    space = LowerTriangularMatrices
    skip_test_basis_belongs = True

    class LowerTriangularMatricesTestData(_VectorSpaceTestData):
        n_list = random.sample(range(2, 5), 2)
        space_args_list = [(n,) for n in n_list]
        shape_list = [(n, n) for n in n_list]
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_test_data(self):
            smoke_data = [
                dict(n=2, mat=[[1.0, 0.0], [-1.0, 3.0]], expected=True),
                dict(n=2, mat=[[1.0, -1.0], [-1.0, 3.0]], expected=False),
                dict(
                    n=2,
                    mat=[
                        [[1.0, 0], [0, 1.0]],
                        [[1.0, 2.0], [2.0, 1.0]],
                        [[-1.0, 0.0], [1.0, 1.0]],
                        [[0.0, 0.0], [1.0, 1.0]],
                    ],
                    expected=[True, False, True, True],
                ),
                dict(
                    n=3,
                    mat=[
                        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[-1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    ],
                    expected=[False, True, True, True],
                ),
                dict(n=3, mat=[[1.0, 0.0], [-1.0, 3.0]], expected=False),
            ]
            return self.generate_tests(smoke_data)

        def random_point_and_belongs_test_data(self):
            smoke_data = [
                dict(n=1, n_points=1),
                dict(n=2, n_points=2),
                dict(n=10, n_points=100),
                dict(n=100, n_points=10),
            ]
            return self.generate_tests(smoke_data)

        def to_vector_test_data(self):
            smoke_data = [
                dict(
                    n=3,
                    mat=[[1.0, 0.0, 0.0], [0.6, 7.0, 0.0], [-3.0, 0.0, 8.0]],
                    expected=[1.0, 0.6, 7.0, -3.0, 0.0, 8.0],
                ),
                dict(
                    n=3,
                    mat=[
                        [[1.0, 0.0, 0.0], [0.6, 7.0, 0.0], [-3.0, 0.0, 8.0]],
                        [[2.0, 0.0, 0.0], [2.6, 7.0, 0.0], [-3.0, 0.0, 28.0]],
                    ],
                    expected=[
                        [1.0, 0.6, 7.0, -3.0, 0.0, 8.0],
                        [2.0, 2.6, 7.0, -3.0, 0.0, 28.0],
                    ],
                ),
            ]
            return self.generate_tests(smoke_data)

        def get_basis_test_data(self):
            smoke_data = [
                dict(
                    n=2,
                    expected=[
                        [[1.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [1.0, 0.0]],
                        [[0.0, 0.0], [0.0, 1.0]],
                    ],
                )
            ]
            return self.generate_tests(smoke_data)

        def projection_test_data(self):
            smoke_data = [
                dict(
                    n=2,
                    point=[[2.0, 1.0], [1.0, 2.0]],
                    expected=[[2.0, 0.0], [1.0, 2.0]],
                ),
                dict(
                    n=2,
                    point=[[1.0, 0.0], [0.0, 1.0]],
                    expected=[[1.0, 0.0], [0.0, 1.0]],
                ),
            ]
            return self.generate_tests(smoke_data)

        def basis_belongs_test_data(self):
            return self._basis_belongs_test_data(self.space_args_list)

        def basis_cardinality_test_data(self):
            return self._basis_cardinality_test_data(self.space_args_list)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(2,), (3,)]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list, self.shape_list, self.n_points_list
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                LowerTriangularMatrices,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                LowerTriangularMatrices, self.space_args_list, self.n_vecs_list
            )

        def to_tangent_is_projection_test_data(self):
            return self._to_tangent_is_projection_test_data(
                LowerTriangularMatrices,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def random_point_is_tangent_test_data(self):
            return self._random_point_is_tangent_test_data(
                self.space_args_list, self.n_points_list
            )

    testing_data = LowerTriangularMatricesTestData()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(self.space(n).belongs(gs.array(mat)), gs.array(expected))

    def test_random_point_and_belongs(self, n, n_points):
        space_n = self.space(n)
        self.assertAllClose(
            gs.all(space_n.belongs(space_n.random_point(n_points))), True
        )

    def test_to_vector(self, n, mat, expected):
        self.assertAllClose(self.space(n).to_vector(gs.array(mat)), gs.array(expected))

    def test_get_basis(self, n, expected):
        self.assertAllClose(self.space(n).basis, gs.array(expected))

    def test_projection(self, n, point, expected):
        self.assertAllClose(
            self.space(n).projection(gs.array(point)), gs.array(expected)
        )
