"""Unit tests for the manifold of matrices."""

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from tests.conftest import Parametrizer, TestCase, autograd_and_torch_only
from tests.data.full_rank_correlation_matrices_data import (
    CorrelationMatricesBundleTestData,
    FullRankcorrelationAffineQuotientMetricTestData,
    FullRankCorrelationMatricesTestData,
)
from tests.geometry_test_cases import LevelSetTestCase


class TestFullRankCorrelationMatrices(LevelSetTestCase, metaclass=Parametrizer):

    skip_test_intrinsic_after_extrinsic = True
    skip_test_extrinsic_after_intrinsic = True
    testing_data = FullRankCorrelationMatricesTestData()


class TestCorrelationMatricesBundle(TestCase, metaclass=Parametrizer):
    testing_data = CorrelationMatricesBundleTestData()
    Space = testing_data.Space
    Base = testing_data.Base

    def test_riemannian_submersion_belongs_to_base(self, n, point):
        bundle = self.Space(n)
        base = self.Base(n)
        result = base.belongs(bundle.riemannian_submersion(point))
        self.assertTrue(gs.all(result))

    def test_lift_riemannian_submersion_composition(self, n, point):
        bundle = self.Space(n)
        result = bundle.riemannian_submersion(bundle.lift(point))
        self.assertAllClose(result, point)

    def test_tangent_riemannian_submersion(self, n, vec, point):
        bundle = self.Space(n)
        tangent_vec = bundle.tangent_riemannian_submersion(vec, point)
        result = gs.all(bundle.is_tangent(tangent_vec, point))
        self.assertTrue(result)

    def test_vertical_projection_tangent_submersion(self, n, vec, mat):
        bundle = self.Space(n)
        tangent_vec = bundle.to_tangent(vec, mat)
        proj = bundle.vertical_projection(tangent_vec, mat)
        result = bundle.tangent_riemannian_submersion(proj, mat)
        expected = gs.zeros_like(vec)
        self.assertAllClose(result, expected)

    def test_horizontal_projection(self, n, vec, mat):
        bundle = self.Space(n)
        base = self.Base(n)
        horizontal_vec = bundle.horizontal_projection(vec, mat)
        inverse = GeneralLinear.inverse(mat)
        product_1 = Matrices.mul(horizontal_vec, inverse)
        product_2 = Matrices.mul(inverse, horizontal_vec)
        is_horizontal = gs.all(
            base.is_tangent(product_1 + product_2, mat, atol=gs.atol * 10)
        )
        self.assertTrue(is_horizontal)

    def test_horizontal_lift_is_horizontal(self, n, tangent_vec, mat):
        bundle = self.Space(n)
        lift = bundle.horizontal_lift(tangent_vec, mat)
        result = gs.all(bundle.is_horizontal(lift, mat))
        self.assertTrue(result)

    def test_vertical_projection_is_vertical(self, n, tangent_vec, mat):
        bundle = self.Space(n)
        proj = bundle.vertical_projection(tangent_vec, mat)
        result = gs.all(bundle.is_vertical(proj, mat))
        self.assertTrue(result)

    @autograd_and_torch_only
    def test_log_after_align_is_horizontal(self, n, point_a, point_b):
        bundle = self.Space(n)
        aligned = bundle.align(point_a, point_b, tol=1e-10)
        log = bundle.metric.log(aligned, point_b)
        result = bundle.is_horizontal(log, point_b, atol=1e-2)
        self.assertTrue(result)

    def test_horizontal_lift_and_tangent_riemannian_submersion(
        self, n, tangent_vec, mat
    ):
        bundle = self.Space(n)
        horizontal = bundle.horizontal_lift(tangent_vec, mat)
        result = bundle.tangent_riemannian_submersion(horizontal, mat)
        self.assertAllClose(result, tangent_vec, atol=gs.atol * 100)


class TestFullRankCorrelationAffineQuotientMetric(TestCase, metaclass=Parametrizer):
    testing_data = FullRankcorrelationAffineQuotientMetricTestData()
    Metric = testing_data.Metric

    @autograd_and_torch_only
    def test_exp_log_composition(self, space, n_points):
        space.equip_with_metric(self.Metric)

        point = space.random_point(n_points)
        base_point = space.random_point(n_points)

        log = space.metric.log(point, base_point)
        result = space.metric.exp(log, base_point)
        self.assertAllClose(result, point, atol=gs.atol * 10000)

    def test_exp_belongs(self, space, n_points):
        space.equip_with_metric(self.Metric)
        bundle = space.metric.fiber_bundle

        base_point = space.random_point(n_points)
        tangent_vec = space.to_tangent(bundle.random_point(), base_point)

        exp = space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(space.belongs(exp), True)
