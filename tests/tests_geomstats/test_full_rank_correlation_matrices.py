"""Unit tests for the manifold of matrices."""

import geomstats.backend as gs
from geomstats.geometry.full_rank_correlation_matrices import (
    CorrelationMatricesBundle,
    FullRankCorrelationAffineQuotientMetric,
    FullRankCorrelationMatrices,
)
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from tests.conftest import Parametrizer, TestCase, autograd_tf_and_torch_only
from tests.data.full_rank_correlation_matrices_data import (
    CorrelationMatricesBundleTestData,
    FullRankcorrelationAffineQuotientMetricTestData,
    RankFullRankCorrelationMatricesTestData,
)
from tests.geometry_test_cases import LevelSetTestCase


class TestFullRankCorrelationMatrices(LevelSetTestCase, metaclass=Parametrizer):

    space = FullRankCorrelationMatrices
    skip_test_intrinsic_after_extrinsic = True
    skip_test_extrinsic_after_intrinsic = True
    testing_data = RankFullRankCorrelationMatricesTestData()


class TestCorrelationMatricesBundle(TestCase, metaclass=Parametrizer):
    space = CorrelationMatricesBundle
    testing_data = CorrelationMatricesBundleTestData()

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
        self.assertAllClose(result, tangent_vec, atol=gs.atol * 100)


class TestFullRankCorrelationAffineQuotientMetric(TestCase, metaclass=Parametrizer):
    metric = connection = FullRankCorrelationAffineQuotientMetric
    testing_data = FullRankcorrelationAffineQuotientMetricTestData()

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
