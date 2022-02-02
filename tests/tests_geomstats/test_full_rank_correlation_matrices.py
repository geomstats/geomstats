"""Unit tests for the manifold of matrices."""

import random

import geomstats.backend as gs
from geomstats.geometry.full_rank_correlation_matrices import (
    CorrelationMatricesBundle,
    FullRankCorrelationAffineQuotientMetric,
    FullRankCorrelationMatrices,
)
from tests.conftest import Parametrizer, TestCase, TestData


class TestFullRankCorrelationMatrices(TestCase, metaclass=Parametrizer):

    cls = FullRankCorrelationMatrices

    class TestDataRankFullRankCorrelationMatrices(TestData):
        def belongs_data(self):
            pass

    def test_belongs(self):
        pass


class TestCorrelationMatricesBundle(TestCase, metaclass=Parametrizer):
    cls = CorrelationMatricesBundle

    class TestDataCorrelationMatricesBundle(TestData):
        def riemannian_submersion_belongs_to_base_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples_list = random.sample(range(1, 50), 10)
            random_data = [
                dict(n=n, mat=CorrelationMatricesBundle(n).random_point(n_samples))
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def lift_riemannian_submersion_composition_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples_list = random.sample(range(1, 50), 10)
            random_data = [
                dict(n=n, mat=CorrelationMatricesBundle(n).random_point(n_samples))
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def tangent_riemannian_submersion_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples_list = random.sample(range(1, 50), 10)
            random_data = [
                dict(
                    n=n,
                    tangent_vec=CorrelationMatricesBundle(n).random_point(n_samples),
                    point=CorrelationMatricesBundle(n).tangent_riemannian_submersion(
                        CorrelationMatricesBundle(n).random_point(n_samples)
                    ),
                )
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def horizontal_lift_is_horizontal_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples_list = random.sample(range(1, 50), 10)
            random_data = [
                dict(
                    n=n,
                    tangent_vec=CorrelationMatricesBundle(n).random_point(n_samples),
                    point=CorrelationMatricesBundle(n).tangent_riemannian_submersion(
                        CorrelationMatricesBundle(n).random_point(n_samples)
                    ),
                )
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def vertical_projection_is_vertical_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples_list = random.sample(range(1, 50), 10)
            random_data = [
                dict(
                    n=n,
                    tangent_vec=CorrelationMatricesBundle(n).random_point(n_samples),
                    point=CorrelationMatricesBundle(n).tangent_riemannian_submersion(
                        CorrelationMatricesBundle(n).random_point(n_samples)
                    ),
                )
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def vertical_projection_tangent_submersion_data(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples_list = random.sample(range(1, 50), 10)
            random_data = [
                dict(
                    n=n,
                    tangent_vec=CorrelationMatricesBundle(n).random_point(n_samples),
                    point=CorrelationMatricesBundle(n).tangent_riemannian_submersion(
                        CorrelationMatricesBundle(n).random_point(n_samples)
                    ),
                )
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

        def horizontal_lift_and_tangent_riemannian_submersion(self):
            n_list = random.sample(range(2, 50), 10)
            n_samples_list = random.sample(range(1, 50), 10)
            random_data = [
                dict(
                    n=n,
                    tangent_vec=CorrelationMatricesBundle(n).random_point(n_samples),
                    point=CorrelationMatricesBundle(n).tangent_riemannian_submersion(
                        CorrelationMatricesBundle(n).random_point(n_samples)
                    ),
                )
                for n, n_samples in zip(n_list, n_samples_list)
            ]
            return self.generate_tests([], random_data)

    def test_riemannian_submersion_belongs_to_base(self, n, mat):
        bundle = self.cls(n)
        result = bundle.base.belongs(bundle.riemannian_submersion(gs.array(mat)))
        self.assertAllClose(result, gs.array(True))

    def lift_riemannian_submersion_composition(self, n, point):
        bundle = self.cls(n)
        result = bundle.riemannian_submersion(bundle.lift(gs.array(point)))
        self.assertAllClose(result, gs.array(point))

    def test_tangent_riemannian_submersion(self, n, tangent_vec, point):
        bundle = self.cls(n)
        vec = bundle.tangent_riemannian_submersion(
            gs.array(tangent_vec), gs.array(point)
        )
        result = gs.all(bundle.is_tangent(vec, gs.array(point)))
        self.assertAllClose(result, gs.array(True))

    def test_vertical_projection_tangent_submersion(
        self, n, tangent_vec, point, expected
    ):
        bundle = self.cls(n)
        proj = bundle.vertical_projection(gs.array(tangent_vec), gs.array(point))
        result = bundle.tangent_riemannian_submersion(proj, gs.array(point))
        self.assertAllClose(result, gs.array(expected))

    def test_horizontal_lift_is_horizontal(self, n, tangent_vec, mat):
        bundle = self.cls(n)
        lift = bundle.horizontal_lift(gs.array(tangent_vec), gs.array(mat))
        result = gs.all(bundle.is_horizontal(lift))
        self.assertAllClose(result, gs.array(True))

    def test_vertical_projection_is_vertical(self, n, tangent_vec, mat):
        bundle = self.cls(n)
        proj = bundle.vertical_projection(gs.array(tangent_vec), gs.array(mat))
        result = gs.all(bundle.is_vertical(proj))
        self.assertAllClose(result, gs.array(True))

    def test_horizontal_lift_and_tangent_riemannian_submersion(
        self, n, tangent_vec, mat
    ):
        bundle = self.cls(n)
        horizontal = bundle.horizontal_lift(gs.array(tangent_vec), gs.array(mat))
        result = bundle.tangent_riemannian_submersion(horizontal, gs.array(mat))
        self.assertAllClose(result, tangent_vec)


class TestFullRankCorrelationAffineQuotientMetric(TestCase, metaclass=Parametrizer):
    cls = FullRankCorrelationAffineQuotientMetric

    class TestDataFullRankcorrelationAffineQuotientMetric(TestData):
        pass
