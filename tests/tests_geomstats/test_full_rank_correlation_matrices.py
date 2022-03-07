"""Unit tests for the manifold of matrices."""

import random

import geomstats.backend as gs
from geomstats.geometry.full_rank_correlation_matrices import (
    CorrelationMatricesBundle,
    FullRankCorrelationAffineQuotientMetric,
    FullRankCorrelationMatrices,
)
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.conftest import TestCase, autograd_tf_and_torch_only
from tests.data_generation import LevelSetTestData, TestData
from tests.parametrizers import LevelSetParametrizer, Parametrizer


class TestFullRankCorrelationMatrices(TestCase, metaclass=LevelSetParametrizer):

    space = FullRankCorrelationMatrices
    skip_test_extrinsic_intrinsic_composition = True
    skip_test_intrinsic_extrinsic_composition = True

    class TestDataRankFullRankCorrelationMatrices(LevelSetTestData):

        n_list = random.sample(range(2, 5), 2)
        space_args_list = [(n,) for n in n_list]
        shape_list = [(n, n) for n in n_list]
        n_samples_list = random.sample(range(2, 5), 2)
        n_points_list = random.sample(range(2, 5), 2)
        n_vecs_list = random.sample(range(2, 5), 2)

        def random_point_belongs_data(self):
            smoke_space_args_list = [(2,), (3,)]
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
                FullRankCorrelationMatrices,
                self.space_args_list,
                self.shape_list,
                self.n_vecs_list,
            )

    testing_data = TestDataRankFullRankCorrelationMatrices()


class TestCorrelationMatricesBundle(TestCase, metaclass=Parametrizer):
    space = CorrelationMatricesBundle

    class TestDataCorrelationMatricesBundle(TestData):
        n_list = random.sample(range(2, 3), 1)
        n_samples_list = random.sample(range(1, 3), 1)

        def riemannian_submersion_belongs_to_base_data(self):
            random_data = []
            for n, n_samples in zip(self.n_list, self.n_samples_list):
                bundle = CorrelationMatricesBundle(n)
                point = bundle.base.random_point(n_samples)
                random_data.append(dict(n=n, point=point))
            return self.generate_tests([], random_data)

        def lift_riemannian_submersion_composition_data(self):
            random_data = []
            for n, n_samples in zip(self.n_list, self.n_samples_list):
                bundle = CorrelationMatricesBundle(n)
                point = bundle.base.random_point(n_samples)
                random_data.append(dict(n=n, point=point))
            return self.generate_tests([], random_data)

        def tangent_riemannian_submersion_data(self):
            random_data = []
            for n, n_samples in zip(self.n_list, self.n_samples_list):
                bundle = CorrelationMatricesBundle(n)
                mat = bundle.random_point()
                point = bundle.riemannian_submersion(mat)
                vec = bundle.random_point(n_samples)
                random_data.append(dict(n=n, vec=vec, point=point))
            return self.generate_tests([], random_data)

        def vertical_projection_tangent_submersion_data(self):
            random_data = []
            for n in self.n_list:
                bundle = CorrelationMatricesBundle(n)
                mat = bundle.random_point(2)
                vec = SymmetricMatrices(n).random_point(2)
                random_data.append(dict(n=n, vec=vec, mat=mat))
            return self.generate_tests([], random_data)

        def horizontal_projection_data(self):
            random_data = []
            for n in self.n_list:
                bundle = CorrelationMatricesBundle(n)
                mat = bundle.random_point()
                vec = bundle.random_point()
                random_data.append(dict(n=n, vec=vec, mat=mat))
            return self.generate_tests([], random_data)

        def horizontal_lift_is_horizontal_data(self):
            random_data = []
            for n, n_samples in zip(self.n_list, self.n_samples_list):
                bundle = CorrelationMatricesBundle(n)
                mat = bundle.base.random_point()
                vec = bundle.base.random_point(n_samples)
                tangent_vec = bundle.base.to_tangent(vec, mat)
                random_data.append(dict(n=n, tangent_vec=tangent_vec, mat=mat))
            return self.generate_tests([], random_data)

        def vertical_projection_is_vertical_data(self):
            random_data = []
            for n, n_samples in zip(self.n_list, self.n_samples_list):
                bundle = CorrelationMatricesBundle(n)
                mat = bundle.random_point()
                vec = bundle.random_point(n_samples)
                tangent_vec = bundle.base.to_tangent(vec, mat)
                random_data.append(dict(n=n, tangent_vec=tangent_vec, mat=mat))
            return self.generate_tests([], random_data)

        def horizontal_lift_and_tangent_riemannian_submersion_data(self):
            random_data = []
            for n, n_samples in zip(self.n_list, self.n_samples_list):
                bundle = CorrelationMatricesBundle(n)
                mat = bundle.base.random_point()
                vec = bundle.base.random_point(n_samples)
                tangent_vec = bundle.base.to_tangent(vec, mat)
                random_data.append(dict(n=n, tangent_vec=tangent_vec, mat=mat))

            return self.generate_tests([], random_data)

        def log_after_align_is_horizontal_data(self):
            n_list = [2, 3]
            random_data = []
            for n in n_list:
                bundle = CorrelationMatricesBundle(n)
                point = bundle.random_point(2)
                random_data.append(dict(n=n, point_a=point[0], point_b=point[1]))
            return self.generate_tests([], random_data)

    testing_data = TestDataCorrelationMatricesBundle()

    def test_riemannian_submersion_belongs_to_base(self, n, point):
        bundle = self.space(n)
        result = bundle.base.belongs(bundle.riemannian_submersion(gs.array(point)))
        self.assertAllClose(gs.all(result), gs.array(True))

    def test_lift_riemannian_submersion_composition(self, n, point):
        bundle = self.space(n)
        result = bundle.riemannian_submersion(bundle.lift(gs.array(point)))
        self.assertAllClose(result, gs.array(point))

    def test_tangent_riemannian_submersion(self, n, vec, point):
        bundle = self.space(n)
        tangent_vec = bundle.tangent_riemannian_submersion(
            gs.array(vec), gs.array(point)
        )
        result = gs.all(bundle.is_tangent(gs.array(tangent_vec), gs.array(point)))
        self.assertAllClose(result, gs.array(True))

    def test_vertical_projection_tangent_submersion(self, n, vec, mat):
        bundle = self.space(n)
        tangent_vec = bundle.to_tangent(vec, mat)
        proj = bundle.vertical_projection(gs.array(tangent_vec), gs.array(mat))
        result = bundle.tangent_riemannian_submersion(proj, gs.array(mat))
        expected = gs.zeros_like(vec)
        self.assertAllClose(result, gs.array(expected))

    def test_horizontal_projection(self, n, vec, mat):
        bundle = self.space(n)
        horizontal_vec = bundle.horizontal_projection(vec, mat)
        inverse = GeneralLinear.inverse(mat)
        product_1 = Matrices.mul(horizontal_vec, inverse)
        product_2 = Matrices.mul(inverse, horizontal_vec)
        is_horizontal = gs.all(
            bundle.base.is_tangent(product_1 + product_2, mat, atol=gs.atol * 10)
        )
        self.assertAllClose(is_horizontal, gs.array(True))

    def test_horizontal_lift_is_horizontal(self, n, tangent_vec, mat):
        bundle = self.space(n)
        lift = bundle.horizontal_lift(gs.array(tangent_vec), gs.array(mat))
        result = gs.all(bundle.is_horizontal(lift, gs.array(mat)))
        self.assertAllClose(result, gs.array(True))

    def test_vertical_projection_is_vertical(self, n, tangent_vec, mat):
        bundle = self.space(n)
        proj = bundle.vertical_projection(gs.array(tangent_vec), gs.array(mat))
        result = gs.all(bundle.is_vertical(proj, gs.array(mat)))
        self.assertAllClose(result, gs.array(True))

    @autograd_tf_and_torch_only
    def test_log_after_align_is_horizontal(self, n, point_a, point_b):
        bundle = self.space(n)
        aligned = bundle.align(point_a, point_b, tol=1e-10)
        log = bundle.ambient_metric.log(aligned, point_b)
        result = bundle.is_horizontal(log, point_b, atol=1e-2)
        self.assertAllClose(result, gs.array(True))

    def test_horizontal_lift_and_tangent_riemannian_submersion(
        self, n, tangent_vec, mat
    ):
        bundle = self.space(n)
        horizontal = bundle.horizontal_lift(gs.array(tangent_vec), gs.array(mat))
        result = bundle.tangent_riemannian_submersion(horizontal, gs.array(mat))
        self.assertAllClose(result, tangent_vec)


class TestFullRankCorrelationAffineQuotientMetric(TestCase, metaclass=Parametrizer):
    metric = connection = FullRankCorrelationAffineQuotientMetric

    class TestDataFullRankcorrelationAffineQuotientMetric(TestData):
        def exp_log_composition_data(self):
            bundle = CorrelationMatricesBundle(3)
            point = bundle.riemannian_submersion(bundle.random_point(2))
            random_data = [dict(dim=3, point=point)]
            return self.generate_tests([], random_data)

        def exp_belongs_data(self):

            bundle = CorrelationMatricesBundle(3)
            base_point = bundle.base.random_point()
            tangent_vec = bundle.base.to_tangent(bundle.random_point(), base_point)
            smoke_data = [dict(dim=3, tangent_vec=tangent_vec, base_point=base_point)]
            return self.generate_tests(smoke_data)

    testing_data = TestDataFullRankcorrelationAffineQuotientMetric()

    @autograd_tf_and_torch_only
    def test_exp_log_composition(self, dim, point):

        metric = self.metric(dim)
        log = metric.log(point[1], point[0])
        result = metric.exp(log, point[0])
        self.assertAllClose(result, point[1], atol=gs.atol * 10000)

    def test_exp_belongs(self, dim, tangent_vec, base_point):
        metric = self.metric(dim)
        exp = metric.exp(tangent_vec, base_point)
        self.assertAllClose(CorrelationMatricesBundle(dim).belongs(exp), True)
