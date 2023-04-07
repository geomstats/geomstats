from tests2.data.base_data import (
    FiberBundleTestData,
    ManifoldTestData,
    _ProjectionMixinsTestData,
)
from tests2.data.full_rank_matrices_data import FullRankMatricesTestData
from tests2.data.quotient_metric_data import QuotientMetricTestData


class RankKPSDMatricesTestData(_ProjectionMixinsTestData, ManifoldTestData):
    tolerances = {
        "to_tangent_is_tangent": {"atol": 1e-1},
    }
    xfails = ("to_tangent_is_tangent",)


class BuresWassersteinBundleTestData(FiberBundleTestData, FullRankMatricesTestData):
    skips = (
        "horizontal_lift_vec",
        "horizontal_lift_is_horizontal",
        # not implemented
        "integrability_tensor_vec",
        "integrability_tensor_derivative_vec",
    )

    xfails = ("tangent_riemannian_submersion_after_horizontal_lift",)


class PSDBuresWassersteinMetricTestData(QuotientMetricTestData):
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
