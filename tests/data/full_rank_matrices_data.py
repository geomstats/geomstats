import random

from geomstats.geometry.full_rank_matrices import FullRankMatrices
from tests.data_generation import _OpenSetTestData


class FullRankMatricesTestData(_OpenSetTestData):
    m_list = random.sample(range(3, 5), 2)
    n_list = random.sample(range(3, 5), 2)
    space_args_list = list(zip(m_list, n_list))
    shape_list = space_args_list
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    space = FullRankMatrices

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
            dict(m=3, n=2, mat=[[1.0, -1.0], [1.0, -1.0], [0.0, 0.0]], expected=False),
        ]
        return self.generate_tests(smoke_data)
