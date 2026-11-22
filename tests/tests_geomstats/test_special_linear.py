"""Unit tests for the special linear group and traceless matrices."""


import random

import geomstats.backend as gs
from geomstats.geometry.special_linear import SpecialLinear, TracelessMatrices
from tests.conftest import TestCase
from tests.data_generation import _LieGroupTestData, _MatrixLieAlgebraTestData

# from tests.parametrizers import LieGroupParametrizer, MatrixLieAlgebraParametrizer


# TODO: very similar to GeneralLinear tests (simplify common points)


class TestSpecialLinear(TestCase):
    space = group = SpecialLinear

    # skips due to tolerance issues
    skip_test_exp_log_composition = True
    skip_test_log_exp_composition = True

    class TestDataSpecialLinear(_LieGroupTestData):
        n_list = random.sample(range(2, 5), 2)
        space_args_list = [(n,) for n in n_list]

        shape_list = [(n, n) for n in n_list]
        n_points_list = random.sample(range(2, 5), 2)
        n_samples_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_data(self):
            smoke_data = [
                dict(n=3, mat=gs.eye(3), expected=True),
                dict(n=3, mat=2 * gs.eye(3), expected=False),
                dict(n=3, mat=-1 * gs.eye(3), expected=False),
                dict(
                    n=2,
                    mat=gs.array(
                        [
                            [gs.cos(gs.pi), gs.sin(gs.pi)],
                            [-gs.sin(gs.pi), gs.cos(gs.pi)],
                        ]
                    ),
                    expected=True,
                ),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_data(self):
            return self._random_point_belongs_data(
                [],
                [],
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_data(self):
            return self._projection_belongs_data(
                self.space_args_list, self.shape_list, self.n_samples_list
            )

        def to_tangent_is_tangent_data(self):
            return self._to_tangent_is_tangent_data(
                SpecialLinear,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def exp_log_composition_data(self):
            return self._exp_log_composition_data(
                SpecialLinear,
                self.space_args_list,
                self.shape_list,
                self.n_samples_list,
            )

        def log_exp_composition_data(self):
            return self._log_exp_composition_data(
                SpecialLinear, self.space_args_list, self.n_samples_list
            )

    testing_data = TestDataSpecialLinear()

    def test_belongs(self, n, point, expected):
        group = self.space(n)
        self.assertAllClose(group.belongs(gs.array(point)), gs.array(expected))


class TestTracelessMatrices(TestCase):
    space = algebra = TracelessMatrices

    class TestDataTracelessMatrices(_MatrixLieAlgebraTestData):
        n_list = random.sample(range(2, 5), 2)
        space_args_list = [(n,) for n in n_list]
        shape_list = [(n, n) for n in n_list]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_data(self):
            smoke_data = [
                dict(n=3, mat=gs.eye(3), expected=False),
                dict(n=2, mat=gs.array([[1, 1], [1, -1]]), expected=True),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_data(self):
            return self._random_point_belongs_data(
                [],
                [],
                self.space_args_list,
                self.n_points_list,
            )

        def projection_belongs_data(self):
            return self._projection_belongs_data(
                self.space_args_list, self.shape_list, self.n_samples_list
            )

        def basis_representation_matrix_representation_composition_data(self):
            return self._basis_representation_matrix_representation_composition_data(
                TracelessMatrices, self.space_args_list, self.n_samples_list
            )

        def matrix_representation_basis_representation_composition_data(self):
            return self._matrix_representation_basis_representation_composition_data(
                TracelessMatrices, self.space_args_list, self.n_samples_list
            )

        def basis_belongs_data(self):
            return self._basis_belongs_data(self.space_args_list)

        def basis_cardinality_data(self):
            return self._basis_cardinality_data(self.space_args_list)

        def to_tangent_is_tangent_data(self):
            return self._to_tangent_is_tangent_data(
                TracelessMatrices,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

    testing_data = TestDataTracelessMatrices()

    def test_belongs(self, n, point, expected):
        group = self.space(n)
        self.assertAllClose(group.belongs(gs.array(point)), gs.array(expected))
