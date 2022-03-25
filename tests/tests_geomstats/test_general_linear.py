"""Unit tests for the General Linear group."""

import random

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear, SquareMatrices
from tests.conftest import Parametrizer
from tests.data_generation import (
    _LieGroupTestData,
    _MatrixLieAlgebraTestData,
    _OpenSetTestData,
)
from tests.geometry_test_cases import (
    LieGroupTestCase,
    MatrixLieAlgebraTestCase,
    OpenSetTestCase,
)


class TestGeneralLinear(LieGroupTestCase, OpenSetTestCase, metaclass=Parametrizer):
    space = group = GeneralLinear
    skip_test_exp_then_log = True
    skip_test_log_then_exp = True

    class GeneralLinearTestData(_LieGroupTestData, _OpenSetTestData):
        n_list = random.sample(range(2, 5), 2)
        positive_det_list = [True, False]
        space_args_list = list(zip(n_list, positive_det_list))
        shape_list = [(n, n) for n in n_list]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_test_data(self):
            smoke_data = [
                dict(n=3, mat=gs.eye(3), expected=True),
                dict(n=3, mat=gs.ones((3, 3)), expected=False),
                dict(n=3, mat=gs.ones(3), expected=False),
                dict(n=3, mat=[gs.eye(3), gs.ones((3, 3))], expected=[True, False]),
            ]
            return self.generate_tests(smoke_data)

        def compose_test_data(self):
            smoke_data = [
                dict(
                    n=2,
                    mat1=[[1.0, 0.0], [0.0, 2.0]],
                    mat2=[[2.0, 0.0], [0.0, 1.0]],
                    expected=2.0 * GeneralLinear(2).identity,
                )
            ]
            return self.generate_tests(smoke_data)

        def inv_test_data(self):
            mat_a = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            smoke_data = [
                dict(
                    n=3,
                    mat=gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]),
                    expected=(
                        1.0
                        / 3.0
                        * gs.array(
                            [[-2.0, -4.0, 3.0], [-2.0, 11.0, -6.0], [3.0, -6.0, 3.0]]
                        )
                    ),
                ),
                dict(
                    n=3,
                    mat=gs.array([mat_a, -gs.eye(3, 3)]),
                    expected=gs.array([mat_a, -gs.eye(3, 3)]),
                ),
            ]
            return self.generate_tests(smoke_data)

        def exp_test_data(self):
            smoke_data = [
                dict(
                    n=3,
                    tangent_vec=[
                        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
                        [[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
                    ],
                    base_point=None,
                    expected=[
                        [
                            [7.38905609, 0.0, 0.0],
                            [0.0, 20.0855369, 0.0],
                            [0.0, 0.0, 54.5981500],
                        ],
                        [
                            [2.718281828, 0.0, 0.0],
                            [0.0, 148.413159, 0.0],
                            [0.0, 0.0, 403.42879349],
                        ],
                    ],
                )
            ]
            return self.generate_tests(smoke_data)

        def log_test_data(self):
            smoke_data = [
                dict(
                    n=3,
                    tangent_vec=[
                        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
                        [[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
                    ],
                    base_point=None,
                    expected=[
                        [
                            [0.693147180, 0.0, 0.0],
                            [0.0, 1.09861228866, 0.0],
                            [0.0, 0.0, 1.38629436],
                        ],
                        [
                            [0.0, 0.0, 0.0],
                            [0.0, 1.609437912, 0.0],
                            [0.0, 0.0, 1.79175946],
                        ],
                    ],
                )
            ]
            return self.generate_tests(smoke_data)

        def orbit_test_data(self):
            point = gs.array([[gs.exp(4.0), 0.0], [0.0, gs.exp(2.0)]])
            sqrt = gs.array([[gs.exp(2.0), 0.0], [0.0, gs.exp(1.0)]])
            identity = GeneralLinear(2).identity
            time = gs.linspace(0.0, 1.0, 3)
            smoke_data = [
                dict(
                    n=2,
                    point=point,
                    base_point=identity,
                    time=time,
                    expected=gs.array([identity, sqrt, point]),
                ),
                dict(
                    n=2,
                    point=[point, point],
                    base_point=identity,
                    time=time,
                    expected=[
                        gs.array([identity, sqrt, point]),
                        gs.array([identity, sqrt, point]),
                    ],
                ),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(2, True), (3, True), (2, False)]
            smoke_n_points_list = [1, 2, 1]
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_test_data(self):
            return self._projection_belongs_test_data(
                self.space_args_list, self.shape_list, self.n_samples_list
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                GeneralLinear,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                GeneralLinear, self.space_args_list, self.n_vecs_list
            )

        def to_tangent_is_tangent_in_ambient_space_test_data(self):
            return self._to_tangent_is_tangent_in_ambient_space_test_data(
                GeneralLinear, self.space_args_list, self.shape_list
            )

        def exp_then_log_test_data(self):
            return self._exp_then_log_test_data(
                GeneralLinear,
                self.space_args_list,
                self.shape_list,
                self.n_samples_list,
                amplitude=10,
                atol=gs.atol * 100000,
            )

        def log_then_exp_test_data(self):
            return self._log_then_exp_test_data(
                GeneralLinear, self.space_args_list, self.n_samples_list, atol=1e-2
            )

    testing_data = GeneralLinearTestData()

    def test_belongs(self, n, point, expected):
        group = self.space(n)
        self.assertAllClose(group.belongs(gs.array(point)), gs.array(expected))

    def test_compose(self, n, mat1, mat2, expected):
        group = self.space(n)
        self.assertAllClose(
            group.compose(gs.array(mat1), gs.array(mat2)), gs.array(expected)
        )

    def test_inv(self, n, mat, expected):
        group = self.space(n)
        self.assertAllClose(group.inverse(gs.array(mat)), gs.array(expected))

    def test_exp(self, n, tangent_vec, base_point, expected):
        group = self.space(n)
        expected = gs.cast(gs.array(expected), gs.float64)
        tangent_vec = gs.cast(gs.array(tangent_vec), gs.float64)
        base_point = (
            None if base_point is None else gs.cast(gs.array(base_point), gs.float64)
        )
        self.assertAllClose(group.exp(tangent_vec, base_point), gs.array(expected))

    def test_log(self, n, point, base_point, expected):
        group = self.space(n)
        expected = gs.cast(gs.array(expected), gs.float64)
        point = gs.cast(gs.array(point), gs.float64)
        base_point = (
            None if base_point is None else gs.cast(gs.array(base_point), gs.float64)
        )
        self.assertAllClose(group.log(point, base_point), expected)

    def test_orbit(self, n, point, base_point, time, expected):
        group = self.space(n)
        result = group.orbit(gs.array(point), gs.array(base_point))(time)
        self.assertAllClose(result, gs.array(expected))


class TestSquareMatrices(MatrixLieAlgebraTestCase, metaclass=Parametrizer):
    space = algebra = SquareMatrices

    class SquareMatricesTestData(_MatrixLieAlgebraTestData):
        n_list = random.sample(range(2, 5), 2)
        space_args_list = [(n,) for n in n_list]
        shape_list = [(n, n) for n in n_list]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_test_data(self):
            smoke_data = [
                dict(n=3, mat=gs.eye(3), expected=True),
                dict(n=3, mat=gs.ones((3, 3)), expected=True),
                dict(n=3, mat=gs.ones(3), expected=False),
            ]
            return self.generate_tests(smoke_data)

        def basis_representation_then_matrix_representation_test_data(self):
            return self._basis_representation_then_matrix_representation_test_data(
                SquareMatrices, self.space_args_list, self.n_samples_list
            )

        def matrix_representation_then_basis_representation_test_data(self):
            return self._matrix_representation_then_basis_representation_test_data(
                SquareMatrices, self.space_args_list, self.n_samples_list
            )

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
                self.space_args_list, self.shape_list, self.n_samples_list
            )

        def to_tangent_is_tangent_test_data(self):
            return self._to_tangent_is_tangent_test_data(
                SquareMatrices,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                SquareMatrices, self.space_args_list, self.n_vecs_list
            )

    testing_data = SquareMatricesTestData()

    def test_belongs(self, n, mat, expected):
        space = self.space(n)
        self.assertAllClose(space.belongs(gs.array(mat)), gs.array(expected))
