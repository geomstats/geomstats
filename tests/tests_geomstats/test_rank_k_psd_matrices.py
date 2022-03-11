r"""Unit tests for the space of PSD matrices of rank k."""

import random

import geomstats.backend as gs
from geomstats.geometry.rank_k_psd_matrices import PSDMatrices
from tests.conftest import TestCase
from tests.data_generation import _ManifoldTestData
from tests.parametrizers import ManifoldParametrizer


class TestPSDMatrices(TestCase, metaclass=ManifoldParametrizer):
    space = PSDMatrices

    class PSDMatricesTestData(_ManifoldTestData):
        n_list = random.sample(range(3, 5), 2)
        k_list = n_list
        space_args_list = list(zip(n_list, k_list))
        shape_list = [(n, n) for n in n_list]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_test_data(self):
            smoke_data = [
                dict(
                    n=3,
                    k=2,
                    mat=[
                        [0.8369314, -0.7342977, 1.0402943],
                        [0.04035992, -0.7218659, 1.0794858],
                        [0.9032698, -0.73601735, -0.36105633],
                    ],
                    expected=False,
                ),
                dict(
                    n=3,
                    k=2,
                    mat=[[1.0, 1.0, 0], [1.0, 4.0, 0], [0, 0, 0]],
                    expected=True,
                ),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(2, 2), (3, 2)]
            smoke_n_points_list = [1, 2]
            belongs_atol = gs.atol * 100000
            return self._random_point_belongs_test_data(
                smoke_space_args_list,
                smoke_n_points_list,
                self.space_args_list,
                self.n_points_list,
                belongs_atol,
            )

        def projection_belongs_test_data(self):
            belongs_atol = gs.atol * 100000
            return self._projection_belongs_test_data(
                self.space_args_list, self.shape_list, self.n_samples_list, belongs_atol
            )

        def to_tangent_is_tangent_test_data(self):
            is_tangent_atol = gs.atol * 100000
            return self._to_tangent_is_tangent_test_data(
                PSDMatrices,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
                is_tangent_atol,
            )

    testing_data = PSDMatricesTestData()

    def test_belongs(self, n, k, mat, expected):
        space = self.space(n, k)
        self.assertAllClose(space.belongs(gs.array(mat)), gs.array(expected))
