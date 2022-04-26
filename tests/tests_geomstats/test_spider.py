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
    skip_all = IS_NOT_NP


class TestSpiderPoint(PointTestCase, metaclass=Parametrizer):
    testing_data = SpiderPointTestData()
    skip_all = IS_NOT_NP


class TestSpiderMetric(PointSetMetricTestCase, metaclass=Parametrizer):
    testing_data = SpiderMetricTestData()
    skip_all = IS_NOT_NP
