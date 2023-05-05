import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.information_geometry.normal import (
    CenteredNormalDistributions,
    CenteredNormalMetric,
    DiagonalNormalDistributions,
    DiagonalNormalMetric,
    GeneralNormalDistributions,
)
from tests.data_generation import _OpenSetTestData, _RiemannianMetricTestData


class CenteredNormalDistributionsTestData(_OpenSetTestData):
    Space = CenteredNormalDistributions
    sample_dim_list = [3, 5, 10]
    space_args_list = [(sample_dim,) for sample_dim in sample_dim_list]
    shape_list = [(sample_dim, sample_dim) for sample_dim in sample_dim_list]
    n_samples_list = [1, 3, 5]
    n_points_list = [1, 3, 5]
    n_vecs_list = [1, 3, 5]

    def belongs_test_data(self):
        sample_dim = self.sample_dim_list[0]
        n_samples = self.n_samples_list[0]
        random_data = [
            dict(
                sample_dim=sample_dim,
                point=self.Space(sample_dim).random_point(n_samples=n_samples),
                expected=gs.array([True] * n_samples),
            )
        ]
        sample_dim = self.sample_dim_list[1]
        n_samples = self.n_samples_list[1]
        random_data.append(
            dict(
                sample_dim=sample_dim,
                point=self.Space(sample_dim).random_point(n_samples=n_samples),
                expected=gs.array([True] * n_samples),
            )
        )
        sample_dim = self.sample_dim_list[2]
        n_samples = self.n_samples_list[2]
        point = gs.random.rand(n_samples, sample_dim, sample_dim)
        random_data.append(
            dict(
                sample_dim=sample_dim,
                point=point,
                expected=gs.array([False] * n_samples),
            )
        )
        return self.generate_tests(random_data)

    def random_point_shape_test_data(self):
        random_data = list()
        for sample_dim, n_samples in zip(self.sample_dim_list, self.n_samples_list):
            if n_samples == 1:
                expected = (sample_dim, sample_dim)
            else:
                expected = (n_samples, sample_dim, sample_dim)
            random_data.append(
                dict(
                    point=self.Space(sample_dim).random_point(n_samples),
                    expected=expected,
                )
            )
        return self.generate_tests(random_data)

    def sample_test_data(self):
        random_data = list()
        for sample_dim, n_samples in zip(self.sample_dim_list, self.n_samples_list):
            for n_points in self.n_points_list:
                if n_samples == 1:
                    if n_points == 1:
                        expected = (sample_dim,)
                    else:
                        expected = (n_points, sample_dim)
                else:
                    if n_points == 1:
                        expected = (n_samples, sample_dim)
                    else:
                        expected = (n_points, n_samples, sample_dim)
                random_data.append(
                    dict(
                        sample_dim=sample_dim,
                        point=self.Space(sample_dim).random_point(n_points),
                        n_samples=n_samples,
                        expected=expected,
                    )
                )
        return self.generate_tests(random_data)

    def point_to_pdf_test_data(self):
        random_data = list()
        for sample_dim, n_samples in zip(self.sample_dim_list, self.n_samples_list):
            for n_points in self.n_points_list:
                random_data.append(
                    dict(
                        sample_dim=sample_dim,
                        point=self.Space(sample_dim).random_point(n_points),
                        n_samples=n_samples,
                    )
                )
        return self.generate_tests([], random_data)


class DiagonalNormalDistributionsTestData(_OpenSetTestData):
    Space = DiagonalNormalDistributions
    sample_dim_list = [3, 5, 10]
    space_args_list = [(sample_dim,) for sample_dim in sample_dim_list]
    shape_list = [(2 * sample_dim,) for sample_dim in sample_dim_list]
    n_samples_list = [1, 3, 5]
    n_points_list = [1, 3, 5]
    n_vecs_list = [1, 3, 5]

    def belongs_test_data(self):
        random_data = list()
        sample_dim = self.sample_dim_list[0]
        n_samples = self.n_samples_list[0]
        random_data = [
            dict(
                sample_dim=sample_dim,
                point=self.Space(sample_dim).random_point(n_samples=n_samples),
                expected=gs.array([True] * n_samples),
            )
        ]
        sample_dim = self.sample_dim_list[1]
        n_samples = self.n_samples_list[1]
        random_data.append(
            dict(
                sample_dim=sample_dim,
                point=self.Space(sample_dim).random_point(n_samples=n_samples),
                expected=gs.array([True] * n_samples),
            )
        )
        sample_dim = self.sample_dim_list[2]
        euc = Euclidean(dim=2 * sample_dim)
        n_samples = self.n_samples_list[2]
        point = euc.random_point(n_samples=n_samples)
        point[-1] = -1
        random_data.append(
            dict(
                sample_dim=sample_dim,
                point=point,
                expected=gs.array([False] * n_samples),
            )
        )
        return self.generate_tests(random_data)

    def random_point_shape_test_data(self):
        random_data = list()
        for sample_dim, n_samples in zip(self.sample_dim_list, self.n_samples_list):
            if n_samples == 1:
                expected = (2 * sample_dim,)
            else:
                expected = (n_samples, 2 * sample_dim)
            random_data.append(
                dict(
                    point=self.Space(sample_dim).random_point(n_samples),
                    expected=expected,
                )
            )
        return self.generate_tests(random_data)

    def sample_test_data(self):
        random_data = list()
        for sample_dim, n_samples in zip(self.sample_dim_list, self.n_samples_list):
            for n_points in self.n_points_list:
                if n_samples == 1:
                    if n_points == 1:
                        expected = (sample_dim,)
                    else:
                        expected = (n_points, sample_dim)
                else:
                    if n_points == 1:
                        expected = (n_samples, sample_dim)
                    else:
                        expected = (n_points, n_samples, sample_dim)
                random_data.append(
                    dict(
                        sample_dim=sample_dim,
                        point=self.Space(sample_dim).random_point(n_points),
                        n_samples=n_samples,
                        expected=expected,
                    )
                )
        return self.generate_tests(random_data)

    def point_to_pdf_test_data(self):
        random_data = list()
        for sample_dim, n_samples in zip(self.sample_dim_list, self.n_samples_list):
            for n_points in self.n_points_list:
                random_data.append(
                    dict(
                        sample_dim=sample_dim,
                        point=self.Space(sample_dim).random_point(n_points),
                        n_samples=n_samples,
                    )
                )
        return self.generate_tests([], random_data)


class GeneralNormalDistributionsTestData(_OpenSetTestData):
    Space = GeneralNormalDistributions
    sample_dim_list = [3, 5, 10]
    space_args_list = [(sample_dim,) for sample_dim in sample_dim_list]
    shape_list = [(sample_dim + sample_dim**2,) for sample_dim in sample_dim_list]
    n_samples_list = [1, 3, 5]
    n_points_list = [1, 3, 5]
    n_vecs_list = [1, 3, 5]

    def unstack_mean_covariance_test_data(self):
        random_data = list()
        for sample_dim, n_samples in zip(self.sample_dim_list, self.n_samples_list):
            if n_samples == 1:
                mean_expected = (sample_dim,)
                cov_expected = (
                    sample_dim,
                    sample_dim,
                )
            else:
                mean_expected = (n_samples, sample_dim)
                cov_expected = (n_samples, sample_dim, sample_dim)
            random_data.append(
                dict(
                    sample_dim=sample_dim,
                    point=self.Space(sample_dim).random_point(n_samples),
                    mean_expected=mean_expected,
                    cov_expected=cov_expected,
                )
            )
        return self.generate_tests(random_data)

    def belongs_test_data(self):
        sample_dim = self.sample_dim_list[0]
        n_samples = self.n_samples_list[0]
        random_data = [
            dict(
                sample_dim=sample_dim,
                point=self.Space(sample_dim).random_point(n_samples=n_samples),
                expected=gs.array([True] * n_samples),
            )
        ]
        sample_dim = self.sample_dim_list[1]
        n_samples = self.n_samples_list[1]
        random_data.append(
            dict(
                sample_dim=sample_dim,
                point=self.Space(sample_dim).random_point(n_samples=n_samples),
                expected=gs.array([True] * n_samples),
            )
        )
        sample_dim = self.sample_dim_list[2]
        n_samples = self.n_samples_list[2]
        point = gs.random.rand(n_samples, sample_dim + sample_dim**2)
        random_data.append(
            dict(
                sample_dim=sample_dim,
                point=point,
                expected=gs.array([False] * n_samples),
            )
        )
        return self.generate_tests(random_data)

    def random_point_shape_test_data(self):
        random_data = list()
        for sample_dim, n_samples in zip(self.sample_dim_list, self.n_samples_list):
            if n_samples == 1:
                expected = (sample_dim + sample_dim**2,)
            else:
                expected = (n_samples, sample_dim + sample_dim**2)
            random_data.append(
                dict(
                    point=self.Space(sample_dim).random_point(n_samples),
                    expected=expected,
                )
            )
        return self.generate_tests(random_data)

    def sample_test_data(self):
        random_data = list()
        for sample_dim, n_samples in zip(self.sample_dim_list, self.n_samples_list):
            for n_points in self.n_points_list:
                if n_samples == 1:
                    if n_points == 1:
                        expected = (sample_dim,)
                    else:
                        expected = (n_points, sample_dim)
                else:
                    if n_points == 1:
                        expected = (n_samples, sample_dim)
                    else:
                        expected = (n_points, n_samples, sample_dim)
                random_data.append(
                    dict(
                        sample_dim=sample_dim,
                        point=self.Space(sample_dim).random_point(n_points),
                        n_samples=n_samples,
                        expected=expected,
                    )
                )
        return self.generate_tests(random_data)

    def point_to_pdf_test_data(self):
        random_data = list()
        for sample_dim, n_samples in zip(self.sample_dim_list, self.n_samples_list):
            for n_points in self.n_points_list:
                random_data.append(
                    dict(
                        sample_dim=sample_dim,
                        point=self.Space(sample_dim).random_point(n_points),
                        n_samples=n_samples,
                    )
                )
        return self.generate_tests([], random_data)


class CenteredNormalMetricTestData(_RiemannianMetricTestData):
    Space = CenteredNormalDistributions
    Metric = CenteredNormalMetric

    sample_dim_list = [3, 5, 10]
    space_list = [
        CenteredNormalDistributions(sample_dim) for sample_dim in sample_dim_list
    ]
    shape_list = [(sample_dim, sample_dim) for sample_dim in sample_dim_list]
    connection_args_list = metric_args_list = [{} for _ in sample_dim_list]

    n_samples_list = [1, 3, 5]
    n_points_list = n_points_a_list = n_points_b_list = [1, 3, 5]
    n_tangent_vecs_list = [1, 3, 5]

    def inner_product_shape_test_data(self):
        random_data = []
        for space, shape, n_tangent_vecs in zip(
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        ):
            batch_shape = () if n_tangent_vecs == 1 else (n_tangent_vecs,)
            base_point = space.random_point()
            tangent_vec_a = space.to_tangent(
                gs.random.normal(scale=1e-2, size=batch_shape + shape), base_point
            )
            tangent_vec_b = space.to_tangent(
                gs.random.normal(scale=1e-2, size=batch_shape + shape), base_point
            )
            expected = () if n_tangent_vecs == 1 else (n_tangent_vecs,)
            random_data.append(
                dict(
                    space=space,
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                )
            )
        return self.generate_tests(random_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=10.0)

    def dist_test_data(self):
        random_data = []
        for space, n_points_a, n_points_b in zip(
            self.space_list,
            self.n_points_a_list,
            self.n_points_b_list,
        ):
            metric = space.metric
            point_a = space.random_point(n_points_a)
            point_b = space.random_point(n_points_b)
            mat = gs.einsum("...ik,...kj->...ij", gs.linalg.inv(point_a), point_b)
            eigenval, _ = gs.linalg.eig(mat)
            expected = (1 / 2 * gs.sum(gs.log(eigenval) ** 2, axis=-1)) ** (1 / 2)
            random_data.append(
                dict(
                    metric=metric,
                    point_a=point_a,
                    point_b=point_b,
                    expected=expected,
                )
            )
        return self.generate_tests(random_data)


class DiagonalNormalMetricTestData(_RiemannianMetricTestData):
    Space = DiagonalNormalDistributions
    Metric = DiagonalNormalMetric

    sample_dim_list = [3, 5, 10]
    space_list = [
        DiagonalNormalDistributions(sample_dim) for sample_dim in sample_dim_list
    ]
    shape_list = [(2 * sample_dim,) for sample_dim in sample_dim_list]
    connection_args_list = metric_args_list = [{} for _ in sample_dim_list]

    n_samples_list = [1, 3, 5]
    n_points_list = n_points_a_list = n_points_b_list = [1, 3, 5]
    n_tangent_vecs_list = [1, 3, 5]

    def inner_product_shape_test_data(self):
        random_data = []
        for space, shape, n_tangent_vecs in zip(
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
        ):
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
                    space=space,
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
        for connection_args, space in zip(self.metric_args_list, self.space_list):
            sample_dim = space.sample_dim
            base_point = 10 * space.random_point()
            tangent_vec = space.to_tangent(0.1 * gs.ones((2 * sample_dim)), base_point)
            random_data.append(
                dict(
                    connection_args=connection_args,
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    atol=gs.atol,
                )
            )
        return self.generate_tests([], random_data)
