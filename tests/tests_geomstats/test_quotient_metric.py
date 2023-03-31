"""Unit tests for the quotient space."""

import pytest

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from tests.conftest import Parametrizer, TestCase
from tests.data.quotient_metric_data import BundleTestData, QuotientMetricTestData


class TestBundle(TestCase, metaclass=Parametrizer):
    testing_data = BundleTestData()
    Bundle = testing_data.Bundle
    Base = testing_data.Base

    def test_riemannian_submersion(self, n):
        bundle = self.Bundle(n)

        mat = bundle.random_point()

        point = bundle.riemannian_submersion(mat)
        result = gs.all(bundle.belongs(point))
        self.assertTrue(result)

    def test_lift_and_riemannian_submersion(self, n):
        bundle = self.Bundle(n)
        base = self.Base(n)

        mat = base.random_point()

        lift = bundle.lift(mat)
        result = bundle.riemannian_submersion(lift)
        self.assertAllClose(result, mat)

    def test_tangent_riemannian_submersion(self, n):
        bundle = self.Bundle(n)

        mat, vec = bundle.random_point(2)

        point = bundle.riemannian_submersion(mat)
        tangent_vec = bundle.tangent_riemannian_submersion(vec, point)
        result = self.Base(n).is_tangent(tangent_vec, point)
        self.assertTrue(result)

    def test_horizontal_projection(self, n):
        bundle = self.Bundle(n)

        mat, vec = bundle.random_point(2)

        horizontal_vec = bundle.horizontal_projection(vec, mat)
        product = Matrices.mul(horizontal_vec, GeneralLinear.inverse(mat))
        is_horizontal = Matrices.is_symmetric(product)
        self.assertTrue(is_horizontal)

    def test_vertical_projection(self, n):
        bundle = self.Bundle(n)

        mat, vec = bundle.random_point(2)

        vertical_vec = bundle.vertical_projection(vec, mat)
        result = bundle.tangent_riemannian_submersion(vertical_vec, mat)
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected, atol=1e-3)

    def test_horizontal_lift_and_tangent_riemannian_submersion(self, n):
        bundle = self.Bundle(n)

        mat, vec = bundle.random_point(2)

        tangent_vec = Matrices.to_symmetric(vec)
        horizontal = bundle.horizontal_lift(tangent_vec, fiber_point=mat)
        result = bundle.tangent_riemannian_submersion(horizontal, mat)
        self.assertAllClose(result, tangent_vec, atol=1e-2)

    @tests.conftest.np_and_autograd_only
    def test_is_horizontal(self, n):
        bundle = self.Bundle(n)

        mat, vec = bundle.random_point(2)

        tangent_vec = Matrices.to_symmetric(vec)
        horizontal = bundle.horizontal_lift(tangent_vec, fiber_point=mat)
        result = bundle.is_horizontal(horizontal, mat, atol=1e-2)
        self.assertTrue(result)

    @tests.conftest.np_and_autograd_only
    def test_is_vertical(self, n):
        bundle = self.Bundle(n)

        mat, vec = bundle.random_point(2)

        vertical = bundle.vertical_projection(vec, mat)
        result = bundle.is_vertical(vertical, mat, atol=1e-2)
        self.assertTrue(result)

    @tests.conftest.autograd_and_torch_only
    def test_align(self, n):
        bundle = self.Bundle(n)

        mat, vec = bundle.random_point(2)

        aligned = bundle.align(mat, vec, tol=1e-10)
        result = bundle.is_horizontal(vec - aligned, vec, atol=1e-2)
        self.assertTrue(result)


class TestQuotientMetric(TestCase, metaclass=Parametrizer):

    testing_data = QuotientMetricTestData()
    Base, Bundle = testing_data.Base, testing_data.Bundle
    ReferenceMetric, Metric = testing_data.ReferenceMetric, testing_data.Metric

    @tests.conftest.np_and_autograd_only
    def test_inner_product(self, n):
        bundle = self.Bundle(n)

        mat, vec_a, vec_b = bundle.random_point(3)

        base = self.Base(n, equip=False)
        base.equip_with_metric(self.ReferenceMetric)

        base_quotient = self.Base(n, equip=False)
        base_quotient.equip_with_metric(self.Metric, fiber_bundle=bundle)

        point = bundle.riemannian_submersion(mat)
        tangent_vecs = Matrices.to_symmetric(gs.array([vec_a, vec_b])) / 40

        result = base_quotient.metric.inner_product(
            tangent_vecs[0], tangent_vecs[1], fiber_point=mat
        )
        expected = base.metric.inner_product(tangent_vecs[0], tangent_vecs[1], point)
        self.assertAllClose(result, expected, atol=1e-1)

    def test_exp(self, n):
        bundle = self.Bundle(n)

        mat, vec = bundle.random_point(2)

        base = self.Base(n, equip=False)
        base.equip_with_metric(self.ReferenceMetric)

        base_quotient = self.Base(n, equip=False)
        base_quotient.equip_with_metric(self.Metric, fiber_bundle=bundle)

        point = bundle.riemannian_submersion(mat)
        tangent_vec = Matrices.to_symmetric(vec) / 40

        result = base_quotient.metric.exp(tangent_vec, point)
        expected = base.metric.exp(tangent_vec, point)
        self.assertAllClose(result, expected, atol=1e-1)

    @tests.conftest.autograd_and_torch_only
    def test_log(self, n):
        bundle = self.Bundle(n)

        mat, vec = bundle.random_point(2)

        base = self.Base(n, equip=False)
        base.equip_with_metric(self.ReferenceMetric)

        base_quotient = self.Base(n, equip=False)
        base_quotient.equip_with_metric(self.Metric, fiber_bundle=bundle)

        points = bundle.riemannian_submersion(gs.array([mat, vec]))

        result = base_quotient.metric.log(points[1], points[0])
        expected = base.metric.log(points[1], points[0])
        self.assertAllClose(result, expected, atol=1e-2)

    @tests.conftest.autograd_and_torch_only
    def test_squared_dist(self, n):
        bundle = self.Bundle(n)

        mat, vec = bundle.random_point(2)

        base = self.Base(n, equip=False)
        base.equip_with_metric(self.ReferenceMetric)

        base_quotient = self.Base(n, equip=False)
        base_quotient.equip_with_metric(self.Metric, fiber_bundle=bundle)

        points = bundle.riemannian_submersion(gs.array([mat, vec]))

        result = base_quotient.metric.squared_dist(points[1], points[0], tol=1e-10)
        expected = base.metric.squared_dist(points[1], points[0])
        self.assertAllClose(result, expected, atol=1e-2)

    def test_integrability_tensor(self, n):
        bundle = self.Bundle(n)

        mat, vec = bundle.random_point(2)

        point = bundle.riemannian_submersion(mat)
        tangent_vec = Matrices.to_symmetric(vec) / 20

        with pytest.raises(NotImplementedError):
            bundle.integrability_tensor(tangent_vec, tangent_vec, point)
