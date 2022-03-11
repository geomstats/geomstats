"""Unit tests for the quotient space."""

import pytest

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricBuresWasserstein
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.conftest import TestCase
from tests.data_generation import TestData
from tests.parametrizers import Parametrizer


class BuresWassersteinBundle(GeneralLinear, FiberBundle):
    def __init__(self, n):
        super(BuresWassersteinBundle, self).__init__(
            n=n,
            base=SPDMatrices(n),
            group=SpecialOrthogonal(n),
            ambient_metric=MatricesMetric(n, n),
        )

    @staticmethod
    def riemannian_submersion(point):
        return Matrices.mul(point, Matrices.transpose(point))

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        product = Matrices.mul(base_point, Matrices.transpose(tangent_vec))
        return 2 * Matrices.to_symmetric(product)

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        if base_point is None:
            if fiber_point is not None:
                base_point = self.riemannian_submersion(fiber_point)
            else:
                raise ValueError(
                    "Either a point (of the total space) or a "
                    "base point (of the base manifold) must be "
                    "given."
                )
        sylvester = gs.linalg.solve_sylvester(base_point, base_point, tangent_vec)
        return Matrices.mul(sylvester, fiber_point)

    @staticmethod
    def lift(point):
        return gs.linalg.cholesky(point)


class TestQuotientMetric(TestCase, metaclass=Parametrizer):
    metric = QuotientMetric
    bundle = BuresWassersteinBundle
    base_metric = SPDMetricBuresWasserstein

    class QuotientMetricTestData(TestData):
        def riemannian_submersion_test_data(self):
            random_data = [dict(n=2, mat=BuresWassersteinBundle(2).random_point())]
            return self.generate_tests([], random_data)

        def lift_and_riemannian_submersion_test_data(self):
            random_data = [dict(n=2, mat=BuresWassersteinBundle(2).base.random_point())]
            return self.generate_tests([], random_data)

        def tangent_riemannian_submersion_test_data(self):
            random_data = [
                dict(
                    n=2,
                    mat=BuresWassersteinBundle(2).random_point(),
                    vec=BuresWassersteinBundle(2).random_point(),
                )
            ]
            return self.generate_tests([], random_data)

        def horizontal_projection_test_data(self):
            return self.tangent_riemannian_submersion_test_data()

        def vertical_projection_test_data(self):
            return self.tangent_riemannian_submersion_test_data()

        def horizontal_lift_and_tangent_riemannian_submersion_test_data(self):
            return self.tangent_riemannian_submersion_test_data()

        def is_horizontal_test_data(self):
            return self.tangent_riemannian_submersion_test_data()

        def is_vertical_test_data(self):
            return self.tangent_riemannian_submersion_test_data()

        def align_test_data(self):
            return self.tangent_riemannian_submersion_test_data()

        def inner_product_test_data(self):
            random_data = [
                dict(
                    n=2,
                    mat=BuresWassersteinBundle(2).random_point(),
                    vec_a=BuresWassersteinBundle(2).random_point(),
                    vec_b=BuresWassersteinBundle(2).random_point(),
                )
            ]
            return self.generate_tests([], random_data)

        def exp_test_data(self):
            return self.tangent_riemannian_submersion_test_data()

        def log_test_data(self):
            return self.tangent_riemannian_submersion_test_data()

        def squared_dist_test_data(self):
            return self.tangent_riemannian_submersion_test_data()

        def integrability_tensor_test_data(self):
            return self.tangent_riemannian_submersion_test_data()

    testing_data = QuotientMetricTestData()

    def test_riemannian_submersion(self, n, mat):
        bundle = self.bundle(n)
        point = bundle.riemannian_submersion(mat)
        result = gs.all(bundle.belongs(point))
        self.assertTrue(result)

    @pytest.mark.skip("giving error")
    def test_lift_and_riemannian_submersion(self, n, mat):
        bundle = self.bundle(n)
        mat = bundle.lift(mat)
        result = bundle.riemannian_submersion(mat)
        self.assertAllClose(result, mat)

    def test_tangent_riemannian_submersion(self, n, mat, vec):
        bundle = self.bundle(n)
        point = bundle.riemannian_submersion(mat)
        tangent_vec = bundle.tangent_riemannian_submersion(vec, point)
        result = bundle.base.is_tangent(tangent_vec, point)
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
        self.assertAllClose(result, tangent_vec, atol=1e-3)

    def test_is_horizontal(self, n, mat, vec):
        bundle = self.bundle(n)
        tangent_vec = Matrices.to_symmetric(vec)
        horizontal = bundle.horizontal_lift(tangent_vec, fiber_point=mat)
        result = bundle.is_horizontal(horizontal, mat, atol=1e-2)
        self.assertTrue(result)

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
