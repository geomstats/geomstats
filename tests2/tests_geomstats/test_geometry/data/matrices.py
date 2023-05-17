from .base import VectorSpaceTestData
from .euclidean import EuclideanMetricTestData


class MatricesTestData(VectorSpaceTestData):
    pass


class MatricesMetricTestData(EuclideanMetricTestData):
    fail_for_not_implemented_errors = False
