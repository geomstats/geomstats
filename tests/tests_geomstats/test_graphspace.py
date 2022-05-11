"""Unit tests for the graphspace quotient space."""


from tests.conftest import Parametrizer
from tests.data.graph_space_data import (
    GraphSpaceMetricTestData,
    GraphSpaceTestData,
    GraphTestData,
)
from tests.stratified_test_cases import (
    PointSetMetricTestCase,
    PointSetTestCase,
    PointTestCase,
)


class TestGraphSpace(PointSetTestCase, metaclass=Parametrizer):
    testing_data = GraphSpaceTestData()


class TestGraphPoint(PointTestCase, metaclass=Parametrizer):
    testing_data = GraphTestData()


class TestGraphSpaceMetric(PointSetMetricTestCase, metaclass=Parametrizer):
    testing_data = GraphSpaceMetricTestData()
    skip_test_geodesic_output_type = True
    skip_test_geodesic = True

    def test_geodesic_graphs(self, space_args, start_point, end_point, t, expected):

        space = self._PointSet(*space_args)

        geom = self._SetGeometry(space)
        geodesic = geom.geodesic(start_point, end_point)
        pts_result = geodesic(t)
        self.assertAllClose(pts_result, expected)

    def test_geodesic_output_type(self, space_args, start_point, end_point):
        space = self._PointSet(*space_args)

        geom = self._SetGeometry(space)
        geodesic = geom.geodesic(start_point, end_point)

        # check output type
        pts_result = geodesic(0.0)
        for pts in pts_result:
            for pt in pts:
                self.assertTrue(type(pt) is self._Point)
