import random

from geomstats.geometry.stratified.spider import Spider
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.stratified.point_set import (
    PointSetMetricTestCase,
    PointSetTestCase,
    PointTestCase,
)

from .data.point_set import PointMetricTestData, PointSetTestData, PointTestData


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
    testing_data = PointMetricTestData()
