from tests2.data.base_data import VectorSpaceTestData
from tests2.data.euclidean_data import EuclideanMetricTestData


class MatricesTestData(VectorSpaceTestData):
    pass


class MatricesMetricTestData(EuclideanMetricTestData):
    skips = (
        # not implemented
        "covariant_riemann_tensor_bianchi_identity",
        "covariant_riemann_tensor_is_interchange_symmetric",
        "covariant_riemann_tensor_is_skew_symmetric_1",
        "covariant_riemann_tensor_is_skew_symmetric_2",
        "covariant_riemann_tensor_vec",
        "curvature_vec",
        "curvature_derivative_vec",
        "directional_curvature_vec",
        "directional_curvature_derivative_vec",
        "injectivity_radius_vec",
        "ricci_tensor_vec",
        "riemann_tensor_vec",
        "scalar_curvature_vec",
        "sectional_curvature_vec",
    )
