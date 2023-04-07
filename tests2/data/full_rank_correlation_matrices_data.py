from tests2.data.base_data import FiberBundleTestData, LevelSetTestData
from tests2.data.quotient_metric_data import QuotientMetricTestData
from tests2.data.spd_matrices_data import SPDMatricesTestData


class FullRankCorrelationMatricesTestData(LevelSetTestData):
    def from_covariance_belongs_test_data(self):
        return self.generate_random_data()

    def from_covariance_vec_test_data(self):
        return self.generate_vec_data()

    def diag_action_vec_test_data(self):
        return self.generate_vec_data()


class CorrelationMatricesBundleTestData(FiberBundleTestData, SPDMatricesTestData):
    xfails = ("align_vec",)

    skips = (
        # not implemented
        "integrability_tensor_vec",
        "integrability_tensor_derivative_vec",
    )
    ignores_if_not_autodiff = (
        "log_after_align_is_horizontal",
        "align_vec",
    )
    tolerances = {
        "log_after_align_is_horizontal": {"atol": 1e-2},
        "align_vec": {"atol": 1e-2},
    }


class FullRankCorrelationAffineQuotientMetricTestData(QuotientMetricTestData):
    xfails = ("log_after_exp",)
    skips = (
        # not implemented
        "christoffels_vec",
        "cometric_matrix_vec",
        "covariant_riemann_tensor_vec",
        "covariant_riemann_tensor_bianchi_identity",
        "covariant_riemann_tensor_is_interchange_symmetric",
        "covariant_riemann_tensor_is_skew_symmetric_1",
        "covariant_riemann_tensor_is_skew_symmetric_2",
        "curvature_vec",
        "curvature_derivative_vec",
        "directional_curvature_vec",
        "directional_curvature_derivative_vec",
        "curvature_derivative_vec",
        "directional_curvature_derivative_vec",
        "injectivity_radius_vec",
        "inner_coproduct_vec",
        "inner_product_derivative_matrix_vec",
        "metric_matrix_is_spd",
        "metric_matrix_vec",
        "ricci_tensor_vec",
        "riemann_tensor_vec",
        "parallel_transport_vec_with_end_point",
        "parallel_transport_vec_with_direction",
        "parallel_transport_transported_is_tangent",
        "scalar_curvature_vec",
        "sectional_curvature_vec",
    )

    tolerances = {
        "dist_point_to_itself_is_zero": {"atol": 1e-6},
        "geodesic_bvp_vec": {"atol": 1e-4},
        "geodesic_bvp_reverse": {"atol": 1e-6},
        "geodesic_boundary_points": {"atol": 1e-6},
        "log_after_exp": {"atol": 1e-4},
        "exp_after_log": {"atol": 1e-6},
        "log_vec": {"atol": 1e-6},
    }
