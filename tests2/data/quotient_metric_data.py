from tests2.data.base_data import RiemannianMetricTestData
from tests2.data.comparison_data import RiemannianMetricComparisonTestData


class QuotientMetricTestData(RiemannianMetricTestData):
    pass


class SPDBuresWassersteinQuotientMetricTestData(RiemannianMetricComparisonTestData):
    skips = (
        # not implemented
        "christoffels",
        "cometric_matrix",
        "covariant_riemann_tensor",
        "curvature_derivative",
        "directional_curvature_derivative",
        "curvature",
        "directional_curvature",
        "inner_coproduct",
        "inner_product_derivative_matrix",
        "metric_matrix",
        "ricci_tensor",
        "riemann_tensor",
        "scalar_curvature",
        "sectional_curvature",
        "parallel_transport_with_direction",
        "parallel_transport_with_end_point",
        "injectivity_radius",
    )
    ignores_if_not_autodiff = (
        "dist",
        "geodesic_bvp",
        "log",
        "squared_dist",
    )

    tolerances = {"geodesic_bvp": {"atol": 1e-6}, "log": {"atol": 1e-6}}
