"""Unit tests for the quotient space."""

import pytest

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricBuresWasserstein
from tests.conftest import Parametrizer, TestCase
from tests.data.quotient_metric_data import (
    BuresWassersteinBundle,
    QuotientMetricTestData,
)


class TestQuotientMetric(TestCase, metaclass=Parametrizer):
    metric = QuotientMetric
    bundle = BuresWassersteinBundle
    base = SPDMatrices
    base_metric = SPDMetricBuresWasserstein

    testing_data = QuotientMetricTestData()

    def test_riemannian_submersion(self, n, mat):
        bundle = self.bundle(n)
        point = bundle.riemannian_submersion(mat)
        result = gs.all(bundle.belongs(point))
        self.assertTrue(result)

    def test_lift_and_riemannian_submersion(self, n, mat):
        bundle = self.bundle(n)
        lift = bundle.lift(mat)
        result = bundle.riemannian_submersion(lift)
        self.assertAllClose(result, mat)

    def test_tangent_riemannian_submersion(self, n, mat, vec):
        bundle = self.bundle(n)
        point = bundle.riemannian_submersion(mat)
        tangent_vec = bundle.tangent_riemannian_submersion(vec, point)
        result = self.base(n).is_tangent(tangent_vec, point)
        self.assertTrue(result)

    def test_horizontal_projection(self, n, mat, vec):
        bundle = self.bundle(n)
        horizontal_vec = bundle.horizontal_projection(vec, mat)
        product = Matrices.mul(horizontal_vec, GeneralLinear.inverse(mat))
        is_horizontal = Matrices.is_symmetric(product)
        self.assertTrue(is_horizontal)

    def test_vertical_projection(self, n, mat, vec):
        bundle = self.bundle(n)
        vertical_vec = bundle.vertical_projection(vec, mat)
        result = bundle.tangent_riemannian_submersion(vertical_vec, mat)
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected, atol=1e-3)

    def test_horizontal_lift_and_tangent_riemannian_submersion(self, n, mat, vec):
        bundle = self.bundle(n)
        tangent_vec = Matrices.to_symmetric(vec)
        horizontal = bundle.horizontal_lift(tangent_vec, fiber_point=mat)
        result = bundle.tangent_riemannian_submersion(horizontal, mat)
        self.assertAllClose(result, tangent_vec, atol=1e-2)

    @geomstats.tests.np_and_autograd_only
    def test_is_horizontal(self, n, mat, vec):
        bundle = self.bundle(n)
        tangent_vec = Matrices.to_symmetric(vec)
        horizontal = bundle.horizontal_lift(tangent_vec, fiber_point=mat)
        result = bundle.is_horizontal(horizontal, mat, atol=1e-2)
        self.assertTrue(result)

    @geomstats.tests.np_and_autograd_only
    def test_is_vertical(self, n, mat, vec):
        bundle = self.bundle(n)
        vertical = bundle.vertical_projection(vec, mat)
        result = bundle.is_vertical(vertical, mat, atol=1e-2)
        self.assertTrue(result)

    @geomstats.tests.autograd_and_torch_only
    def test_align(self, n, point_a, point_b):
        bundle = self.bundle(n)
        aligned = bundle.align(point_a, point_b, tol=1e-10)
        result = bundle.is_horizontal(point_b - aligned, point_b, atol=1e-2)
        self.assertTrue(result)

    @geomstats.tests.np_and_autograd_only
    def test_inner_product(self, n, mat, vec_a, vec_b):
        bundle = self.bundle(n)
        quotient_metric = self.metric(bundle)
        base_metric = self.base_metric(n)
        point = bundle.riemannian_submersion(mat)
        tangent_vecs = Matrices.to_symmetric(gs.array([vec_a, vec_b])) / 40
        result = quotient_metric.inner_product(
            tangent_vecs[0], tangent_vecs[1], fiber_point=mat
        )
        expected = base_metric.inner_product(tangent_vecs[0], tangent_vecs[1], point)
        self.assertAllClose(result, expected, atol=1e-1)

    @geomstats.tests.np_autograd_and_torch_only
    def test_exp(self, n, mat, vec):
        bundle = self.bundle(n)
        quotient_metric = self.metric(bundle)
        base_metric = self.base_metric(n)
        point = bundle.riemannian_submersion(mat)
        tangent_vec = Matrices.to_symmetric(vec) / 40

        result = quotient_metric.exp(tangent_vec, point)
        expected = base_metric.exp(tangent_vec, point)
        self.assertAllClose(result, expected, atol=1e-1)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_log(self, n, mat_a, mat_b):
        bundle = self.bundle(n)
        quotient_metric = self.metric(bundle)
        base_metric = self.base_metric(n)
        points = bundle.riemannian_submersion(gs.array([mat_a, mat_b]))

        result = quotient_metric.log(points[1], points[0])
        expected = base_metric.log(points[1], points[0])
        self.assertAllClose(result, expected, atol=1e-2)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_squared_dist(self, n, mat_a, mat_b):
        bundle = self.bundle(n)
        quotient_metric = self.metric(bundle)
        base_metric = self.base_metric(n)
        points = bundle.riemannian_submersion(gs.array([mat_a, mat_b]))

        result = quotient_metric.squared_dist(points[1], points[0], tol=1e-10)
        expected = base_metric.squared_dist(points[1], points[0])
        self.assertAllClose(result, expected, atol=1e-2)

    def test_integrability_tensor(self, n, mat, vec):
        bundle = self.bundle(n)
        point = bundle.riemannian_submersion(mat)
        tangent_vec = Matrices.to_symmetric(vec) / 20

        with pytest.raises(NotImplementedError):
            bundle.integrability_tensor(tangent_vec, tangent_vec, point)
