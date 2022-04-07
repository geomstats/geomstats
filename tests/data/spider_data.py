
import random

import geomstats.backend as gs
from geomstats.stratified_geometry.spider import SpiderPoint

from tests.data_generation import (
    _PointSetTestData,
    _PointTestData,
    _PointGeometryTestData,
)


class SpiderTestData(_PointSetTestData):

    # for random tests
    n_samples = _PointSetTestData.n_samples
    rays_list = random.sample(range(1, 5), n_samples)
    space_args_list = [(rays,) for rays in rays_list]

    def belongs_test_data(self):
        pt1 = SpiderPoint(3, 13)
        pt2 = SpiderPoint(0, 0)
        pt3 = SpiderPoint(4, 1)

        smoke_data = [
            dict(space_args=(10,), points=pt1, expected=[True]),
            dict(space_args=(0,), points=[pt2, pt3], expected=[True, False]),
            dict(space_args=(2,), points=[pt3], expected=[False])
        ]

        return self.generate_tests(smoke_data)

    def set_to_array_test_data(self):
        pt0 = SpiderPoint(0, 0.)
        pt1, pt2 = SpiderPoint(1, 2.), SpiderPoint(3, 3.)

        smoke_data = [
            dict(space_args=(3,),
                 points=pt0,
                 expected=gs.array([[0., 0., 0.]]),),
            dict(space_args=(3,),
                 points=pt1,
                 expected=gs.array([[2., 0., 0.]]),),
            dict(space_args=(3,),
                 points=[pt1],
                 expected=gs.array([[2., 0., 0.]]),),
            dict(space_args=(3,),
                 points=[pt1, pt2],
                 expected=gs.array([[2., 0, 0.], [0., 0., 3.]]))

        ]

        return self.generate_tests(smoke_data)


class SpiderPointTestData(_PointTestData):

    def to_array_test_data(self):
        smoke_data = [
            dict(point_args=(1, 2.), expected=gs.array([1., 2.])),
            dict(point_args=(2, 1.), expected=gs.array([2., 1.])),
        ]

        return self.generate_tests(smoke_data)


class SpiderGeometryTestData(_PointGeometryTestData):

    def dist_test_data(self):
        pts_start = [SpiderPoint(10, 1.), SpiderPoint(10, 2.),
                     SpiderPoint(3, 1.)]

        pts_end = [SpiderPoint(10, 31.), SpiderPoint(10, 2.),
                   SpiderPoint(1, 4.)]

        smoke_data = [
            dict(space_args=(12,),
                 start_point=pts_start[0],
                 end_point=pts_end[0],
                 expected=gs.array([30.])),
            dict(space_args=(12,),
                 start_point=pts_start[0],
                 end_point=pts_end,
                 expected=gs.array([30., 1., 5.])),
            dict(space_args=(12,),
                 start_point=pts_start,
                 end_point=pts_end[0],
                 expected=gs.array([30., 29., 32.])
                 ),
            dict(space_args=(12,),
                 start_point=pts_start,
                 end_point=pts_end,
                 expected=gs.array([30., 0., 5.])
                 )

        ]

        return self.generate_tests(smoke_data)

    def geodesic_output_type_test_data(self):
        smoke_data = [
            dict(space_args=(12,),
                 start_point=SpiderPoint(1, 2.),
                 end_point=SpiderPoint(2, 3.)),
        ]

        return self.generate_tests(smoke_data)

    def geodesic_test_data(self):
        pts_start = [SpiderPoint(10, 1.), SpiderPoint(10, 2.),
                     SpiderPoint(3, 1.)]

        pts_end = [SpiderPoint(10, 31.), SpiderPoint(10, 2.),
                   SpiderPoint(1, 4.)]

        smoke_data = [
            dict(space_args=(12,),
                 start_point=pts_start[0],
                 end_point=pts_end[0],
                 t=0.,
                 expected=gs.array([[[10., 1.]]])),
            dict(space_args=(12,),
                 start_point=pts_start[0],
                 end_point=pts_end[0],
                 t=0.5,
                 expected=gs.array([[[10., 16.]]])),
            dict(space_args=(12,),
                 start_point=pts_start[0],
                 end_point=pts_end[0],
                 t=[0.],
                 expected=gs.array([[[10., 1.]]])),
            dict(space_args=(12,),
                 start_point=pts_start[0],
                 end_point=pts_end[0],
                 t=[0., 1.],
                 expected=gs.array([[[10., 1.]], [[10., 31.]]])),
            dict(space_args=(12,),
                 start_point=pts_start[0],
                 end_point=pts_end,
                 t=1.,
                 expected=gs.array([[[10., 31.], [10., 2.], [1., 4.]]])),
            dict(space_args=(12,),
                 start_point=pts_start[0],
                 end_point=pts_end,
                 t=[0., 1.],
                 expected=gs.array([
                     [[10., 1], [10., 1.], [10., 1.]],
                     [[10., 31.], [10., 2.], [1., 4.]],
                 ])),
            dict(space_args=(12,),
                 start_point=pts_start,
                 end_point=pts_end,
                 t=[0., 1.],
                 expected=gs.array([
                     [[10., 1], [10., 2.], [3., 1.]],
                     [[10., 31.], [10., 2.], [1., 4.]],
                 ])),

        ]

        return self.generate_tests(smoke_data)
