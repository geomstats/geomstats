import random

from geomstats.metric_geometry.spider import Spider
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.metric_geometry.point_set import (
    PointSetMetricTestCase,
    PointSetTestCase,
    PointTestCase,
)

from .data.point_set import PointSetMetricTestData, PointSetTestData, PointTestData


class TestSpiderPoint(PointTestCase, metaclass=DataBasedParametrizer):
    _n_rays = random.randint(2, 4)
    space = Spider(_n_rays, equip=False)
    testing_data = PointTestData()


class TestSpider(PointSetTestCase, metaclass=DataBasedParametrizer):
    _n_rays = random.randint(2, 4)
    space = Spider(_n_rays, equip=False)
    testing_data = PointSetTestData()


class TestSpiderMetric(PointSetMetricTestCase, metaclass=DataBasedParametrizer):
    _n_rays = random.randint(2, 4)
    space = Spider(_n_rays, equip=True)
    testing_data = PointSetMetricTestData()
