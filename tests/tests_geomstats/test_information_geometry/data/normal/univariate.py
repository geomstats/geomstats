from ....test_geometry.data.poincare_half_space import PoincareHalfSpaceTestData
from ....test_geometry.data.riemannian_metric import (
    CoordinateRiemannianMetricTestData,
)
from ..base import InformationManifoldMixinTestData


class UnivariateNormalDistributionsTestData(
    InformationManifoldMixinTestData, PoincareHalfSpaceTestData
):
    def point_to_pdf_against_scipy_test_data(self):
        return self._generate_random_data_with_samples()


class UnivariateNormalMetricTestData(CoordinateRiemannianMetricTestData):
    def dist_against_closed_form_test_data(self):
        return self.generate_random_data(arg_names=("point_a", "point_b"))

    def sectional_curvature_value_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )
