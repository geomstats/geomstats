from .base import VectorSpaceTestData
from .euclidean import EuclideanMetricTestData


class MatricesTestData(VectorSpaceTestData):
    pass


class MatricesMetricTestData(EuclideanMetricTestData):
    fail_for_not_implemented_errors = False
    skips = (
        "christoffels_vec",
        "inner_coproduct_vec",
        "inner_product_derivative_matrix_vec",
    )
