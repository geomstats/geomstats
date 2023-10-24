import pytest

from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import np_backend
from geomstats.test_cases.geometry.stratified.point_set import (
    PointSetMetricTestCase,
    PointSetTestCase,
    PointTestCase,
)

from .data.spider import SpiderMetricTestData, SpiderPointTestData, SpiderTestData

IS_NOT_NP = not np_backend()

# TODO: update to new format


class TestSpider(PointSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = SpiderTestData()
    skip_all = IS_NOT_NP


class TestSpiderPoint(PointTestCase, metaclass=DataBasedParametrizer):
    testing_data = SpiderPointTestData()
    skip_all = IS_NOT_NP

    def test_raise_zero_error(self, point_args):
        with pytest.raises(ValueError):
            self.testing_data._Point(*point_args)


class TestSpiderMetric(PointSetMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = SpiderMetricTestData()
    skip_all = IS_NOT_NP
