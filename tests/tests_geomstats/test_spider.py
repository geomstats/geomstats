from geomstats.stratified_geometry.spider import Spider, SpiderGeometry, SpiderPoint
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

    _PointSet = Spider
    testing_data = SpiderTestData()


class TestSpiderPoint(PointTestCase, metaclass=Parametrizer):

    _Point = SpiderPoint
    testing_data = SpiderPointTestData()


class TestSpiderGeometry(PointSetGeometryTestCase, metaclass=Parametrizer):

    _SetGeometry = SpiderGeometry
    _PointSet = Spider
    _Point = SpiderPoint
    testing_data = SpiderGeometryTestData()

    def test_geodesic_output_type(self, space_args, start_point, end_point):
        space = self._PointSet(*space_args)

        geom = self._SetGeometry(space)
        geodesic = geom.geodesic(start_point, end_point)

        # check output type
        pts_result = geodesic(0.0)
        for pts in pts_result:
            for pt in pts:
                self.assertTrue(type(pt) is self._Point)
