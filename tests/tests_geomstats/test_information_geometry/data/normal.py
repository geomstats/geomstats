from ...test_geometry.data.base import VectorSpaceOpenSetTestData
from ...test_geometry.data.poincare_half_space import PoincareHalfSpaceTestData
from ...test_geometry.data.product_manifold import ProductManifoldTestData
from ...test_geometry.data.riemannian_metric import RiemannianMetricTestData
from .base import InformationManifoldMixinTestData


class UnivariateNormalDistributionsTestData(
    InformationManifoldMixinTestData, PoincareHalfSpaceTestData
):
    def point_to_pdf_against_scipy_test_data(self):
        return self._generate_random_data_with_samples()


class UnivariateNormalMetricTestData(RiemannianMetricTestData):
    def dist_against_closed_form_test_data(self):
        return self.generate_random_data(arg_names=("point_a", "point_b"))

    def sectional_curvature_value_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )


class CenteredNormalDistributionsTestData(
    InformationManifoldMixinTestData, VectorSpaceOpenSetTestData
):
    def point_to_pdf_against_scipy_test_data(self):
        return self._generate_random_data_with_samples()


class DiagonalNormalDistributionsTestData(
    InformationManifoldMixinTestData, VectorSpaceOpenSetTestData
):
    def point_to_pdf_against_scipy_test_data(self):
        return self._generate_random_data_with_samples()


class GeneralNormalDistributionsTestData(
    InformationManifoldMixinTestData, ProductManifoldTestData
):
    def point_to_pdf_against_scipy_test_data(self):
        return self._generate_random_data_with_samples()
