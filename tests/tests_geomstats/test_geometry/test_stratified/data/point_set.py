from geomstats.test.data import TestData

from ...data.mixins import DistMixinsTestData, GeodesicBVPMixinsTestData


class PointSetTestData(TestData):
    def random_point_belongs_test_data(self):
        random_data = [
            dict(space_args=space_args, n_points=n_points)
            for space_args, n_points in zip(self.space_args_list, self.n_points_list)
        ]

        return self.generate_tests([], random_data)

    def random_point_output_shape_test_data(self):
        space = self._PointSet(*self.space_args_list[0])

        smoke_data = [
            dict(space=space, n_samples=1),
            dict(space=space, n_samples=2),
        ]

        return self.generate_tests(smoke_data)

    def set_to_array_output_shape_test_data(self):
        space = self._PointSet(*self.space_args_list[0])
        pts = space.random_point(2)

        smoke_data = [
            dict(space=space, points=pts[0]),
            dict(space=space, points=pts),
        ]

        return self.generate_tests(smoke_data)


class PointTestData(TestData):
    pass


class PointMetricTestData(TestData):
    def dist_output_shape_test_data(self):
        space = self._PointSet(*self.space_args_list[0], equip=True)
        pts = space.random_point(2)

        dist_fnc = space.metric.dist

        smoke_data = [
            dict(dist_fnc=dist_fnc, point_a=pts[0], point_b=pts[1]),
            dict(dist_fnc=dist_fnc, point_a=pts[0], point_b=pts),
            dict(dist_fnc=dist_fnc, point_a=pts, point_b=pts[0]),
            dict(dist_fnc=dist_fnc, point_a=pts, point_b=pts),
        ]

        return self.generate_tests(smoke_data)

    def dist_properties_test_data(self):
        space = self._PointSet(*self.space_args_list[0], equip=True)
        pts = space.random_point(3)

        dist_fnc = space.metric.dist

        smoke_data = [
            dict(dist_fnc=dist_fnc, point_a=pts[0], point_b=pts[1], point_c=pts[2]),
        ]

        return self.generate_tests(smoke_data)

    def geodesic_output_shape_test_data(self):
        space = self._PointSet(*self.space_args_list[0])
        metric = space.metric
        pts = space.random_point(2)

        smoke_data = [
            dict(metric=metric, start_point=pts[0], end_point=pts[0], t=0.0),
            dict(metric=metric, start_point=pts[0], end_point=pts[0], t=[0.0, 1.0]),
            dict(metric=metric, start_point=pts[0], end_point=pts, t=0.0),
            dict(metric=metric, start_point=pts[0], end_point=pts, t=[0.0, 1.0]),
            dict(metric=metric, start_point=pts, end_point=pts[0], t=0.0),
            dict(metric=metric, start_point=pts, end_point=pts[0], t=[0.0, 1.0]),
            dict(metric=metric, start_point=pts, end_point=pts, t=0.0),
            dict(metric=metric, start_point=pts, end_point=pts, t=[0.0, 1.0]),
        ]

        return self.generate_tests(smoke_data)

    def geodesic_bounds_test_data(self):
        space = self._PointSet(*self.space_args_list[0], equip=True)
        pts = space.random_point(2)

        smoke_data = [dict(space=space, start_point=pts[0], end_point=pts[1])]

        return self.generate_tests(smoke_data)


class PointSetMetricWithArrayTestData(
    DistMixinsTestData,
    GeodesicBVPMixinsTestData,
    TestData,
):
    pass
