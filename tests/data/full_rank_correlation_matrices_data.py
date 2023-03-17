import random

from geomstats.geometry.full_rank_correlation_matrices import (
    CorrelationMatricesBundle,
    FullRankCorrelationMatrices,
)
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.data_generation import TestData, _LevelSetTestData


class FullRankCorrelationMatricesTestData(_LevelSetTestData):

    n_list = random.sample(range(2, 4), 2)
    space_args_list = [(n,) for n in n_list]
    shape_list = [(n, n) for n in n_list]
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = FullRankCorrelationMatrices


class CorrelationMatricesBundleTestData(TestData):
    n_list = random.sample(range(2, 3), 1)
    n_samples_list = random.sample(range(1, 3), 1)

    Space = CorrelationMatricesBundle
    Base = FullRankCorrelationMatrices

    def riemannian_submersion_belongs_to_base_test_data(self):
        random_data = []
        for n, n_samples in zip(self.n_list, self.n_samples_list):
            base = self.Base(n)
            point = base.random_point(n_samples)
            random_data.append(dict(n=n, point=point))
        return self.generate_tests([], random_data)

    def lift_riemannian_submersion_composition_test_data(self):
        random_data = []
        for n, n_samples in zip(self.n_list, self.n_samples_list):
            base = self.Base(n)
            point = base.random_point(n_samples)
            random_data.append(dict(n=n, point=point))
        return self.generate_tests([], random_data)

    def tangent_riemannian_submersion_test_data(self):
        random_data = []
        for n, n_samples in zip(self.n_list, self.n_samples_list):
            bundle = self.Space(n)
            mat = bundle.random_point()
            point = bundle.riemannian_submersion(mat)
            vec = bundle.random_point(n_samples)
            random_data.append(dict(n=n, vec=vec, point=point))
        return self.generate_tests([], random_data)

    def vertical_projection_tangent_submersion_test_data(self):
        random_data = []
        for n in self.n_list:
            bundle = self.Space(n)
            mat = bundle.random_point(2)
            vec = SymmetricMatrices(n).random_point(2)
            random_data.append(dict(n=n, vec=vec, mat=mat))
        return self.generate_tests([], random_data)

    def horizontal_projection_test_data(self):
        random_data = []
        for n in self.n_list:
            bundle = self.Space(n)
            mat = bundle.random_point()
            vec = bundle.random_point()
            random_data.append(dict(n=n, vec=vec, mat=mat))
        return self.generate_tests([], random_data)

    def horizontal_lift_is_horizontal_test_data(self):
        random_data = []
        for n, n_samples in zip(self.n_list, self.n_samples_list):
            base = self.Base(n)
            mat = base.random_point()
            vec = base.random_point(n_samples)
            tangent_vec = base.to_tangent(vec, mat)
            random_data.append(dict(n=n, tangent_vec=tangent_vec, mat=mat))
        return self.generate_tests([], random_data)

    def vertical_projection_is_vertical_test_data(self):
        random_data = []
        for n, n_samples in zip(self.n_list, self.n_samples_list):
            bundle = self.Space(n)
            base = self.Base(n)
            mat = bundle.random_point()
            vec = bundle.random_point(n_samples)
            tangent_vec = base.to_tangent(vec, mat)
            random_data.append(dict(n=n, tangent_vec=tangent_vec, mat=mat))
        return self.generate_tests([], random_data)

    def horizontal_lift_and_tangent_riemannian_submersion_test_data(self):
        random_data = []
        for n, n_samples in zip(self.n_list, self.n_samples_list):
            base = self.Base(n)
            mat = base.random_point()
            vec = base.random_point(n_samples)
            tangent_vec = base.to_tangent(vec, mat)
            random_data.append(dict(n=n, tangent_vec=tangent_vec, mat=mat))

        return self.generate_tests([], random_data)

    def log_after_align_is_horizontal_test_data(self):
        n_list = [2, 3]
        random_data = []
        for n in n_list:
            bundle = self.Space(n)
            point = bundle.random_point(2)
            random_data.append(dict(n=n, point_a=point[0], point_b=point[1]))
        return self.generate_tests([], random_data)


class FullRankcorrelationAffineQuotientMetricTestData(TestData):

    Metric = QuotientMetric
    Space = CorrelationMatricesBundle

    def exp_log_composition_test_data(self):
        bundle = CorrelationMatricesBundle(3)
        point = bundle.riemannian_submersion(bundle.random_point(2))
        random_data = [dict(dim=3, point=point)]
        return self.generate_tests([], random_data)

    def exp_belongs_test_data(self):
        bundle = CorrelationMatricesBundle(3)
        base = FullRankCorrelationMatrices(3)
        base_point = base.random_point()
        tangent_vec = base.to_tangent(bundle.random_point(), base_point)
        smoke_data = [dict(dim=3, tangent_vec=tangent_vec, base_point=base_point)]
        return self.generate_tests(smoke_data)
