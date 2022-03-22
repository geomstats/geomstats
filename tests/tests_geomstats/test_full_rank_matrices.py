"""Unit tests for full rank matrices."""
import random

import geomstats.backend as gs
from geomstats.geometry.full_rank_matrices import FullRankMatrices
from tests.conftest import Parametrizer
from tests.data_generation import _OpenSetTestData
from tests.geometry_test_cases import OpenSetTestCase


class TestFullRankMatrices(OpenSetTestCase, metaclass=Parametrizer):

    space = FullRankMatrices

    class FullRankMatricesTestData(_OpenSetTestData):
        m_list = random.sample(range(3, 5), 2)
        n_list = random.sample(range(3, 5), 2)
        space_args_list = list(zip(m_list, n_list))
        shape_list = space_args_list
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def belongs_test_data(self):
            smoke_data = [
                dict(
                    m=3,
                    n=2,
                    mat=[
                        [-1.6473486, -1.18240309],
                        [0.1944016, 0.18169231],
                        [-1.13933855, -0.64971248],
                    ],
                    expected=True,
                ),
                dict(
                    m=3, n=2, mat=[[1.0, -1.0], [1.0, -1.0], [0.0, 0.0]], expected=False
                ),
            ]
            return self.generate_tests(smoke_data)

        def random_point_belongs_test_data(self):
            smoke_space_args_list = [(2, 2), (3, 2)]
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
                FullRankMatrices,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

        def to_tangent_is_tangent_in_ambient_space_test_data(self):
            return self._to_tangent_is_tangent_in_ambient_space_test_data(
                FullRankMatrices, self.space_args_list, self.shape_list
            )

        def random_tangent_vec_is_tangent_test_data(self):
            return self._random_tangent_vec_is_tangent_test_data(
                FullRankMatrices,
                self.space_args_list,
                self.n_vecs_list,
                is_tangent_atol=gs.atol * 100,
            )

    testing_data = FullRankMatricesTestData()

    def test_belongs(self, m, n, mat, expected):
        self.assertAllClose(self.space(m, n).belongs(gs.array(mat)), gs.array(expected))
