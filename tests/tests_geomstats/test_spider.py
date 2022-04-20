from tests.conftest import Parametrizer, np_backend
from tests.data.spider_data import (
    SpiderMetricTestData,
    SpiderPointTestData,
    SpiderTestData,
)
from tests.stratified_test_cases import (
    PointSetMetricTestCase,
    PointSetTestCase,
    PointTestCase,
)

IS_NOT_NP = not np_backend()


class TestSpider(PointSetTestCase, metaclass=Parametrizer):
    testing_data = SpiderTestData()
    skip_test_set_to_array_output_shape = IS_NOT_NP


class TestSpiderPoint(PointTestCase, metaclass=Parametrizer):
    testing_data = SpiderPointTestData()


class TestSpiderMetric(PointSetMetricTestCase, metaclass=Parametrizer):
    testing_data = SpiderMetricTestData()

    skip_test_dist_output_shape = IS_NOT_NP
    skip_test_dist_properties = IS_NOT_NP
    skip_test_geodesic_bounds = IS_NOT_NP
    skip_test_geodesic_output_shape = IS_NOT_NP
