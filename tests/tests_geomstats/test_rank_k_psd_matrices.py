r"""Unit tests for the space of PSD matrices of rank k."""

import random

import geomstats.backend as gs
import tests.helper as helper
from geomstats.geometry.rank_k_psd_matrices import PSDMatrices
from tests.conftest import Parametrizer, TestCase, TestData


class TestPSDMatrices(TestCase, metaclass=Parametrizer):
    cls = PSDMatrices

    class TestDataPSDMatrices(TestData):
        def belongs_data(self):
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

        def projection_and_belongs_data(self):
            smoke_data = [
                dict(n=2, k=2, n_samples=10),
                dict(n=2, k=1, n_samples=1),
                dict(n=3, k=3, n_samples=5),
            ]
            random_data = []
            n_list = random.sample(range(2, 50), 5)
            for n in n_list:
                k_list = random.sample(range(1, n), 5)
                for k in k_list:
                    n_samples = random.sample(range(2, 50))
                    random_data += [dict(n=n, k=k, n_samples=n_samples)]

            return self.generate_tests(smoke_data, random_data)

        def to_tangent_is_tangent_data(self):
            random_data = []
            n_list = random.sample(range(2, 50), 5)
            for n in n_list:
                k_list = random.sample(range(1, n), 5)
                for k in k_list:
                    n_samples = random.sample(range(1, 50))
                    base_point = PSDMatrices(n, k).random_point(n_samples)
                    mat = gs.random.normal(size=(n_samples, n, n))
                    random_data += [dict(n=n, k=k, mat=mat, base_point=base_point)]

            return self.generate_tests([], random_data)

    def test_belongs(self, n, k, mat, expected):
        space = self.cls(n, k)
        self.assertAllClose(space.belongs(gs.array(mat)), gs.array(expected))

    def test_projection_and_belongs(self, n, k, n_samples):
        group = self.cls(n, k)
        shape = (n_samples, n, n)
        result = helper.test_projection_and_belongs(group, shape)
        self.assertAllClose(gs.all(result), gs.array(True))
