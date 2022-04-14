from tests.conftest import Parametrizer
from tests.data.spider_data import (
    SpiderGeometryTestData,
    SpiderPointTestData,
    SpiderTestData,
)
from tests.stratified_geometry_test_cases import (
    PointSetGeometryTestCase,
    PointSetTestCase,
    PointTestCase,
)


class TestSpider(PointSetTestCase, metaclass=Parametrizer):
    testing_data = SpiderTestData()


class TestSpiderPoint(PointTestCase, metaclass=Parametrizer):
    testing_data = SpiderPointTestData()


class TestSpiderGeometry(PointSetGeometryTestCase, metaclass=Parametrizer):
    testing_data = SpiderGeometryTestData()
