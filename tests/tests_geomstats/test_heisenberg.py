"""Unit tests for the 3D heisenberg group in vector representation."""

import random

import geomstats.backend as gs
from geomstats.geometry.heisenberg import HeisenbergVectors
from tests.conftest import TestCase
from tests.data_generation import LieGroupTestData
from tests.parametrizers import LieGroupParametrizer


class TestHeisenbergVectors(TestCase, metaclass=LieGroupParametrizer):
    space = group = HeisenbergVectors

    class TestDataHeisenbergVectors(LieGroupTestData):
        space_args_list = [()] * 3
        shape_list = [(3,)] * 3
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def dimension_data(self):
            smoke_data = [dict(expected=3)]
            return self.generate_tests(smoke_data)

        def belongs_data(self):
            smoke_data = [
                dict(point=[1.0, 2.0, 3.0, 4], expected=False),
                dict(
                    point=[[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]],
                    expected=[False, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def is_tangent_data(self):
            smoke_data = [
                dict(
                    vec=[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
                    expected=[False, False],
                )
            ]
            return self.generate_tests(smoke_data)

        def jacobian_translation_data(self):
            smoke_data = [
                dict(
                    vec=[[1.0, -10.0, 0.2], [-2.0, 100.0, 0.5]],
                    expected=[
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 0.5, 1.0]],
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-50.0, -1.0, 1.0]],
                    ],
                )
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_data(self):
            smoke_space_args_list = [()] * 2
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
                HeisenbergVectors,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def exp_log_composition_data(self):
            return self._exp_log_composition_data(
                HeisenbergVectors,
                self.space_args_list,
                self.shape_list,
                self.n_samples_list,
            )

        def log_exp_composition_data(self):
            return self._log_exp_composition_data(
                HeisenbergVectors, self.space_args_list, self.n_samples_list
            )

    testing_data = TestDataHeisenbergVectors()

    def test_dimension(self, expected):
        self.assertAllClose(self.space().dim, expected)

    def test_jacobian_translation(self, vec, expected):
        self.assertAllClose(
            self.space().jacobian_translation(gs.array(vec)), gs.array(expected)
        )

    def test_random_point_belongs(self, n_samples, bound):
        self.assertAllClose(gs.all(self.space().random_point(n_samples, bound)), True)

    def test_is_tangent(self, vector, expected):
        group = self.space()
        result = group.is_tangent(gs.array(vector))
        self.assertAllClose(result, gs.array(expected))
