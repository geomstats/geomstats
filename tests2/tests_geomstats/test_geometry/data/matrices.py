from .base import VectorSpaceTestData
from .euclidean import EuclideanMetricTestData


class MatricesTestData(VectorSpaceTestData):
    pass


class MatricesMetricTestData(EuclideanMetricTestData):
    fail_for_not_implemented_errors = False
    skips = (
        "christoffels_vec",
        "cometric_matrix_vec",
        "inner_coproduct_vec",
        "inner_product_derivative_matrix_vec",
        "metric_matrix_is_spd",
        "metric_matrix_vec",
    )
