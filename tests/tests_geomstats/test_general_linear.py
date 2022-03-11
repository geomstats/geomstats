"""Unit tests for the General Linear group."""

import random

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear, SquareMatrices
from tests.conftest import TestCase
from tests.data_generation import LieGroupTestData, MatrixLieAlgebraTestData
from tests.parametrizers import LieGroupParametrizer, MatrixLieAlgebraParametrizer


class TestGeneralLinear(TestCase, metaclass=LieGroupParametrizer):
    space = group = GeneralLinear
    skip_test_exp_log_composition = True
    skip_test_log_exp_composition = True

    class TestDataGeneralLinear(LieGroupTestData):
        n_list = random.sample(range(2, 5), 2)
        positive_det_list = [True, False]
        space_args_list = list(zip(n_list, positive_det_list))
        shape_list = [(n, n) for n in n_list]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_data(self):
            smoke_data = [
                dict(n=3, mat=gs.eye(3), expected=True),
                dict(n=3, mat=gs.ones((3, 3)), expected=False),
                dict(n=3, mat=gs.ones(3), expected=False),
                dict(n=3, mat=[gs.eye(3), gs.ones((3, 3))], expected=[True, False]),
            ]
            return self.generate_tests(smoke_data)

        def compose_data(self):
            smoke_data = [
                dict(
                    n=2,
                    mat1=[[1.0, 0.0], [0.0, 2.0]],
                    mat2=[[2.0, 0.0], [0.0, 1.0]],
                    expected=2.0 * GeneralLinear(2).identity,
                )
            ]
            return self.generate_tests(smoke_data)

        def inv_data(self):
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

        def exp_data(self):
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

        def log_data(self):
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

        def orbit_data(self):
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

        def random_point_belongs_data(self):
            smoke_space_args_list = [(2, True), (3, True), (2, False)]
            smoke_n_points_list = [1, 2, 1]
            return self._random_point_belongs_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_data(self):
            return self._projection_belongs_data(
                self.space_args_list, self.shape_list, self.n_samples_list
            )

        def to_tangent_is_tangent_data(self):
            return self._to_tangent_is_tangent_data(
                GeneralLinear,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def exp_log_composition_data(self):
            return self._exp_log_composition_data(
                GeneralLinear,
                self.space_args_list,
                self.shape_list,
                self.n_samples_list,
            )

        def log_exp_composition_data(self):
            return self._log_exp_composition_data(
                GeneralLinear, self.space_args_list, self.n_samples_list
            )

    testing_data = TestDataGeneralLinear()

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


class TestSquareMatrices(TestCase, metaclass=MatrixLieAlgebraParametrizer):
    space = algebra = SquareMatrices

    class TestDataSquareMatrices(MatrixLieAlgebraTestData):
        n_list = random.sample(range(2, 5), 2)
        space_args_list = [(n,) for n in n_list]
        shape_list = [(n, n) for n in n_list]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_data(self):
            smoke_data = [
                dict(n=3, mat=gs.eye(3), expected=True),
                dict(n=3, mat=gs.ones((3, 3)), expected=True),
                dict(n=3, mat=gs.ones(3), expected=False),
            ]
            return self.generate_tests(smoke_data)

        def basis_representation_matrix_representation_composition_data(self):
            return self._basis_representation_matrix_representation_composition_data(
                SquareMatrices, self.space_args_list, self.n_samples_list
            )

        def matrix_representation_basis_representation_composition_data(self):
            return self._matrix_representation_basis_representation_composition_data(
                SquareMatrices, self.space_args_list, self.n_samples_list
            )

        def basis_belongs_data(self):
            return self._basis_belongs_data(self.space_args_list)

        def basis_cardinality_data(self):
            return self._basis_cardinality_data(self.space_args_list)

        def random_point_belongs_data(self):
            smoke_space_args_list = [(2,), (3,)]
            smoke_n_points_list = [1, 2]
            return self._random_point_belongs_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_data(self):
            return self._projection_belongs_data(
                self.space_args_list, self.shape_list, self.n_samples_list
            )

        def to_tangent_is_tangent_data(self):
            return self._to_tangent_is_tangent_data(
                SquareMatrices,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

    testing_data = TestDataSquareMatrices()

    def test_belongs(self, n, mat, expected):
        space = self.space(n)
        self.assertAllClose(space.belongs(gs.array(mat)), gs.array(expected))
