import random

from polpo.testing.parametrizers import DataBasedParametrizer

from geomstats.metric_geometry.spider import Spider
from geomstats.test_cases.metric_geometry.point_set import (
    PointSetMetricTestCase,
    PointSetTestCase,
    PointTestCase,
)

from .data.point_set import PointSetMetricTestData, PointSetTestData, PointTestData


class TestSpiderPoint(PointTestCase, metaclass=DataBasedParametrizer):
    space = Spider(random.randint(2, 4), equip=False)
    testing_data = PointTestData(space)


class TestSpider(PointSetTestCase, metaclass=DataBasedParametrizer):
    space = Spider(random.randint(2, 4), equip=False)
    testing_data = PointSetTestData(space)


class TestSpiderMetric(PointSetMetricTestCase, metaclass=DataBasedParametrizer):
    space = Spider(random.randint(2, 4))
    testing_data = PointSetMetricTestData(space)
