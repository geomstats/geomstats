import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.information_geometry.multivariate_normal import (
    MultivariateCenteredNormalDistributions,
    MultivariateDiagonalNormalDistributions,
    MultivariateDiagonalNormalMetric,
    MultivariateGeneralNormalDistributions,
)
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData


class MultivariateCenteredNormalDistributionsTestData(_OpenSetTestData):
    Space = MultivariateCenteredNormalDistributions
    n_list = [3, 5, 10]
    space_args_list = [(n,) for n in n_list]
    shape_list = [(n, n) for n in n_list]
    n_samples_list = [1, 3, 5]
    n_points_list = [1, 3, 5]
    n_vecs_list = [1, 3, 5]

    def belongs_test_data(self):
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
        n_samples = self.n_samples_list[2]
        point = gs.random.rand(n_samples, n, n)
        random_data.append(
            dict(n=n, point=point, expected=gs.array([False] * n_samples))
        )
        return self.generate_tests(random_data)

    def random_point_shape_test_data(self):
        random_data = list()
        for n, n_samples in zip(self.n_list, self.n_samples_list):
            expected = (n, n) if n_samples == 1 else (n_samples, n, n)
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

    def point_to_pdf_test_data(self):
        random_data = list()
        for n, n_samples in zip(self.n_list, self.n_samples_list):
            for n_points in self.n_points_list:
                random_data.append(
                    dict(
                        n=n,
                        point=self.Space(n).random_point(n_points),
                        n_samples=n_samples,
                    )
                )
        return self.generate_tests([], random_data)


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

    def point_to_pdf_test_data(self):
        random_data = list()
        for n, n_samples in zip(self.n_list, self.n_samples_list):
            for n_points in self.n_points_list:
                random_data.append(
                    dict(
                        n=n,
                        point=self.Space(n).random_point(n_points),
                        n_samples=n_samples,
                    )
                )
        return self.generate_tests([], random_data)


class MultivariateGeneralNormalDistributionsTestData(_OpenSetTestData):
    Space = MultivariateGeneralNormalDistributions
    n_list = [3, 5, 10]
    space_args_list = [(n,) for n in n_list]
    shape_list = [(n + n**2,) for n in n_list]
    n_samples_list = [1, 3, 5]
    n_points_list = [1, 3, 5]
    n_vecs_list = [1, 3, 5]

    def belongs_test_data(self):
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
        n_samples = self.n_samples_list[2]
        point = gs.random.rand(n_samples, n, n)
        random_data.append(
            dict(n=n, point=point, expected=gs.array([False] * n_samples))
        )
        return self.generate_tests(random_data)

    def random_point_shape_test_data(self):
        random_data = list()
        for n, n_samples in zip(self.n_list, self.n_samples_list):
            expected = (n + n**2,) if n_samples == 1 else (n_samples, n + n**2)
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

    def point_to_pdf_test_data(self):
        random_data = list()
        for n, n_samples in zip(self.n_list, self.n_samples_list):
            for n_points in self.n_points_list:
                random_data.append(
                    dict(
                        n=n,
                        point=self.Space(n).random_point(n_points),
                        n_samples=n_samples,
                    )
                )
        return self.generate_tests([], random_data)


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

    def inner_product_shape_test_data(self):
        random_data = []
        for space, shape, n_tangent_vecs in zip(
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        ):
            metric = space.metric
            base_point = space.random_point()
            tangent_vec_a = space.to_tangent(
                gs.random.normal(scale=1e-2, size=(n_tangent_vecs,) + shape), base_point
            )
            tangent_vec_b = space.to_tangent(
                gs.random.normal(scale=1e-2, size=(n_tangent_vecs,) + shape), base_point
            )
            expected = (tangent_vec_a.shape[0],)
            random_data.append(
                dict(
                    metric=metric,
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                )
            )
        return self.generate_tests(random_data)

    def exp_belongs_test_data(self):
        random_data = []
        for connection_args, space, shape, n_tangent_vecs in zip(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        ):
            base_point = space.random_point()
            tangent_vec = space.to_tangent(
                gs.random.normal(scale=1, size=(n_tangent_vecs,) + shape), base_point
            )
            random_data.append(
                dict(
                    connection_args=connection_args,
                    space=space,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    atol=0,
                )
            )
        return self.generate_tests([], random_data)

    def log_after_exp_test_data(self):
        random_data = []
        for connection_args, space, shape, n_tangent_vecs in zip(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        ):
            n = space.n
            base_point = 10 * space.random_point()
            tangent_vec = space.to_tangent(0.1 * gs.ones((2 * n)), base_point)
            random_data.append(
                dict(
                    connection_args=connection_args,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    atol=gs.atol,
                )
            )
        return self.generate_tests([], random_data)
