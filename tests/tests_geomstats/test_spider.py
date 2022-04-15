from tests.conftest import Parametrizer
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


class TestSpider(PointSetTestCase, metaclass=Parametrizer):
    testing_data = SpiderTestData()


class TestSpiderPoint(PointTestCase, metaclass=Parametrizer):
    testing_data = SpiderPointTestData()


class TestSpiderMetric(PointSetMetricTestCase, metaclass=Parametrizer):
    testing_data = SpiderMetricTestData()
