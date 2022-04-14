import random

import geomstats.backend as gs
from geomstats.geometry.stratified.spider import Spider, SpiderMetric, SpiderPoint
from tests.data_generation import (
    _PointGeometryTestData,
    _PointSetTestData,
    _PointTestData,
)


class SpiderTestData(_PointSetTestData):

    _Point = SpiderPoint
    _PointSet = Spider

    # for random tests
    n_samples = _PointSetTestData.n_samples
    rays_list = random.sample(range(1, 5), n_samples)
    space_args_list = [(rays,) for rays in rays_list]

    def belongs_test_data(self):
        pt1 = self._Point(3, 13)
        pt2 = self._Point(0, 0)
        pt3 = self._Point(4, 1)

        smoke_data = [
            dict(space_args=(10,), points=pt1, expected=[True]),
            dict(space_args=(0,), points=[pt2, pt3], expected=[True, False]),
            dict(space_args=(2,), points=[pt3], expected=[False]),
        ]

        return self.generate_tests(smoke_data)

    def set_to_array_test_data(self):

        pt0 = self._Point(0, 0.0)
        pts = [self._Point(1, 2.0), self._Point(3, 3.0)]

        smoke_data = [
            dict(
                space_args=(3,),
                points=pt0,
                expected=gs.array([[0.0, 0.0, 0.0]]),
            ),
            dict(
                space_args=(3,),
                points=pts,
                expected=gs.array([[2.0, 0, 0.0], [0.0, 0.0, 3.0]]),
            ),
        ]

        return self.generate_tests(smoke_data)


class SpiderPointTestData(_PointTestData):

    _Point = SpiderPoint

    def to_array_test_data(self):
        smoke_data = [
            dict(point_args=(1, 2.0), expected=gs.array([1.0, 2.0])),
            dict(point_args=(2, 1.0), expected=gs.array([2.0, 1.0])),
        ]

        return self.generate_tests(smoke_data)


class SpiderGeometryTestData(_PointGeometryTestData):

    _SetGeometry = SpiderMetric
    _PointSet = Spider
    _Point = SpiderPoint

    # for random tests
    n_samples = _PointSetTestData.n_samples
    rays_list = random.sample(range(1, 5), n_samples)
    space_args_list = [(rays,) for rays in rays_list]

    def dist_test_data(self):
        pts_start = [self._Point(10, 1.0), self._Point(3, 1.0)]
        pts_end = [SpiderPoint(10, 31.0), SpiderPoint(1, 4.0)]

        smoke_data = [
            dict(
                space_args=(12,),
                point_a=pts_start,
                point_b=pts_end,
                expected=gs.array([30.0, 5.0]),
            ),
        ]

        return self.generate_tests(smoke_data)

    def geodesic_test_data(self):

        smoke_data = [
            dict(
                space_args=(12,),
                start_point=self._Point(10, 1.0),
                end_point=self._Point(10, 31.0),
                t=0.4,
                expected=gs.array([[10.0, 13.0]]),
            ),
        ]

        return self.generate_tests(smoke_data)
