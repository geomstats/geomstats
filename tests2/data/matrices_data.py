from tests2.data.base_data import VectorSpaceTestData
from tests2.data.euclidean_data import EuclideanMetricTestData


class MatricesTestData(VectorSpaceTestData):
    pass


class MatricesMetricTestData(EuclideanMetricTestData):
    fail_for_not_implemented_errors = False
