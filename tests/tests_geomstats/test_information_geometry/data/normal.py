from ...test_geometry.data.base import VectorSpaceOpenSetTestData
from ...test_geometry.data.poincare_half_space import PoincareHalfSpaceTestData
from ...test_geometry.data.product_manifold import (
    ProductManifoldTestData,
    ProductRiemannianMetricTestData,
)
from ...test_geometry.data.pullback_metric import PullbackDiffeoMetricTestData
from ...test_geometry.data.riemannian_metric import RiemannianMetricTestData
from ...test_geometry.data.spd_matrices import SPDMatricesTestData
from .base import InformationManifoldMixinTestData


class UnivariateNormalDistributionsTestData(
    InformationManifoldMixinTestData, PoincareHalfSpaceTestData
):
    fail_for_not_implemented_errors = False

    def point_to_pdf_against_scipy_test_data(self):
        return self.generate_random_data_with_samples()


class UnivariateNormalMetricTestData(PullbackDiffeoMetricTestData):
    trials = 4

    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def dist_against_closed_form_test_data(self):
        return self.generate_random_data()

    def sectional_curvature_value_test_data(self):
        return self.generate_random_data()


class CenteredNormalDistributionsTestData(
    InformationManifoldMixinTestData, SPDMatricesTestData
):
    fail_for_not_implemented_errors = False

    def point_to_pdf_against_scipy_test_data(self):
        return self.generate_random_data_with_samples()


class DiagonalNormalDistributionsTestData(
    InformationManifoldMixinTestData, VectorSpaceOpenSetTestData
):
    fail_for_not_implemented_errors = False

    def point_to_pdf_against_scipy_test_data(self):
        return self.generate_random_data_with_samples()


class DiagonalNormalMetricTestData(RiemannianMetricTestData):
    trials = 5
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {"geodesic_ivp_belongs": {"atol": 1e-3}}


class GeneralNormalDistributionsTestData(
    InformationManifoldMixinTestData,
    ProductManifoldTestData,
):
    fail_for_not_implemented_errors = False

    skips = ("not_belongs",)

    def point_to_pdf_against_scipy_test_data(self):
        return self.generate_random_data_with_samples()


class GeneralNormalMetricTestData(ProductRiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
