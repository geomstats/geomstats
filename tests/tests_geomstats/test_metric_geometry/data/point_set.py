from polpo.testing.geometric import GeometricCaseData

from ...test_geometry.data.mixins import DistMixinsTestData, GeodesicBVPMixinsTestData


class PointTestData(GeometricCaseData):
    def point_is_equal_to_itself_test_data(self):
        return self.generate_random_data(arg_names="point")


class PointSetTestData(GeometricCaseData):
    def random_point_belongs_test_data(self):
        return self.generate_random_data(arg_names="point")


class PointSetMetricTestData(DistMixinsTestData, GeometricCaseData):
    def geodesic_boundary_points_test_data(self):
        return self.generate_random_data(arg_names=("initial_point", "end_point"))

    def geodesic_bvp_reverse_test_data(self):
        return self.generate_timed_random_data(
            arg_names=("initial_point", "end_point")
        )


class PointSetMetricWithArrayTestData(
    DistMixinsTestData,
    GeodesicBVPMixinsTestData,
    GeometricCaseData,
):
    pass
