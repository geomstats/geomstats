import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.information_geometry.multivariate_normal import (
    MultivariateDiagonalNormalDistributions,
    MultivariateDiagonalNormalMetric,
)
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData


class MultivariateDiagonalNormalDistributionsTestData(_OpenSetTestData):
    Space = MultivariateDiagonalNormalDistributions
    n_list = [3, 5, 10]
    space_args_list = [(n,) for n in n_list]
    shape_list = [(2 * n,) for n in n_list]
    n_samples_list = [1, 3, 5]
    n_points_list = [1, 3, 5]
    n_vecs_list = [1, 3, 5]

    def belongs_test_data(self):
        random_data = list()
        n = self.n_list[0]
        n_samples = self.n_samples_list[0]
        random_data = [
            dict(
                n=n,
                point=self.Space(n).random_point(n_samples=n_samples),
                expected=gs.array([True] * n_samples),
            )
        ]
        n = self.n_list[1]
        n_samples = self.n_samples_list[1]
        random_data.append(
            dict(
                n=n,
                point=self.Space(n).random_point(n_samples=n_samples),
                expected=gs.array([True] * n_samples),
            )
        )
        n = self.n_list[2]
        euc = Euclidean(dim=2 * n)
        n_samples = self.n_samples_list[2]
        point = euc.random_point(n_samples=n_samples)
        point[-1] = -1
        random_data.append(
            dict(n=n, point=point, expected=gs.array([False] * n_samples))
        )
        return self.generate_tests(random_data)

    def random_point_shape_test_data(self):
        random_data = list()
        for n, n_samples in zip(self.n_list, self.n_samples_list):
            if n_samples == 1:
                expected = (2 * n,)
            else:
                expected = (n_samples, 2 * n)
            random_data.append(
                dict(point=self.Space(n).random_point(n_samples), expected=expected)
            )
        return self.generate_tests(random_data)

    def sample_test_data(self):
        random_data = list()
        for n, n_samples in zip(self.n_list, self.n_samples_list):
            for n_points in self.n_points_list:
                if n_samples == 1:
                    if n_points == 1:
                        expected = (n,)
                    else:
                        expected = (n_points, n)
                else:
                    if n_points == 1:
                        expected = (n_samples, n)
                    else:
                        expected = (n_points, n_samples, n)
                random_data.append(
                    dict(
                        n=n,
                        point=self.Space(n).random_point(n_points),
                        n_samples=n_samples,
                        expected=expected,
                    )
                )
        return self.generate_tests(random_data)


class MultivariateDiagonalNormalMetricTestData(_RiemannianMetricTestData):
    Space = MultivariateDiagonalNormalDistributions
    Metric = MultivariateDiagonalNormalMetric

    n_list = [3, 5, 10]
    space_list = [MultivariateDiagonalNormalDistributions(n) for n in n_list]
    shape_list = [(2 * n,) for n in n_list]
    space_args_list = [(n,) for n in n_list]
    connection_args_list = [(2 * n,) for n in n_list]
    n_samples_list = [1, 3, 5]
    n_points_list = n_points_a_list = n_points_b_list = [1, 3, 5]
    n_tangent_vecs_list = [1, 3, 5]
    metric_args_list = list(
        zip(
            n_list,
        )
    )
