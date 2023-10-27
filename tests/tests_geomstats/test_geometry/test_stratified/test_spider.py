import pytest

from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.stratified.point_set import (
    PointSetMetricTestCase,
    PointSetTestCase,
    PointTestCase,
)

from .data.spider import SpiderMetricTestData, SpiderPointTestData, SpiderTestData


class TestSpider(PointSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = SpiderTestData()


class TestSpiderPoint(PointTestCase, metaclass=DataBasedParametrizer):
    testing_data = SpiderPointTestData()

    def test_raise_zero_error(self, point_args):
        with pytest.raises(ValueError):
            self.testing_data._Point(*point_args)


class TestSpiderMetric(PointSetMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = SpiderMetricTestData()
