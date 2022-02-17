"""Unit tests for full rank matrices."""

import geomstats.backend as gs
from geomstats.geometry.full_rank_matrices import FullRankMatrices
from tests.conftest import TestCase
from tests.data_generation import TemporaryTestData
from tests.parametrizers import Parametrizer


class TestFullRankMatrices(TestCase, metaclass=Parametrizer):

    cls = FullRankMatrices

    class TestDataFullRankMatrices(TemporaryTestData):
        def belongs_data(self):
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

        def random_and_belongs_data(self):
            smoke_data = [
                dict(m=1, n=1, n_points=1),
                dict(m=1, n=1, n_points=1000),
                dict(m=2, n=2, n_points=1),
                dict(m=2, n=2, n_points=100),
                dict(m=10, n=5, n_points=100),
            ]
            return self.generate_tests(smoke_data)

        def projection_and_belongs_data(self):
            shapes = [(1, 1), (1, 1), (1, 10), (2, 2), (10, 5), (15, 15)]
            sizes = [1, 10, 1, 1, 100, 10]
            random_data = [
                dict(m=m, n=n, mats=gs.random.normal(size=(size, m, n)))
                for (m, n), size in zip(shapes, sizes)
            ]

            return self.generate_tests([], random_data)

    testing_data = TestDataFullRankMatrices()

    def test_belongs(self, m, n, mat, expected):
        self.assertAllClose(self.cls(m, n).belongs(gs.array(mat)), gs.array(expected))

    def test_random_and_belongs(self, m, n, n_points):
        cls = self.cls(m, n)
        self.assertAllClose(
            gs.all(cls.belongs(cls.random_point(n_points))), gs.array(True)
        )

    def test_projection_and_belongs(self, m, n, mat):
        self.assertAllClose(gs.all(self.cls(m, n).belongs(mat)), True)
