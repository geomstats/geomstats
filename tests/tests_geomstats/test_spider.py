
import geomstats.backend as gs

from geomstats.stratified_geometry.spider import (
    Spider,
    SpiderPoint,
    SpiderGeometry
)

from tests.conftest import Parametrizer
from tests.stratified_geometry_test_cases import (
    PointSetTestCase,
    PointTestCase,
    PointSetGeometryTestCase,
)
from tests.data.spider_data import (
    SpiderTestData,
    SpiderPointTestData,
    SpiderGeometryTestData,
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

    def test_geodesic(self, space_args, start_point, end_point, t, expected):
        # TODO: generalize? (point type may be tricky)
        # TODO: split testing point?

        space = self._PointSet(*space_args)

        geom = self._SetGeometry(space)
        geodesic = geom.geodesic(start_point, end_point)

        # check output type
        pts_result = geodesic(t)
        for pts in pts_result:
            for pt in pts:
                self.assertTrue(type(pt) is self._Point)

        # check results
        results = []
        for pts in pts_result:
            t_results = []
            for pt in pts:
                t_results.append(pt.to_array())

            results.append(t_results)

        results = gs.array(results)
        self.assertAllClose(results, expected)
