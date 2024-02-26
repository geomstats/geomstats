from geomstats.test.data import TestData

from ...data.mixins import DistMixinsTestData, GeodesicBVPMixinsTestData


class PointTestData(TestData):
    def point_is_equal_to_itself_test_data(self):
        return self.generate_random_data()


class PointSetTestData(TestData):
    def random_point_belongs_test_data(self):
        return self.generate_random_data()


class PointMetricTestData(DistMixinsTestData, TestData):
    skip_vec = True

    def geodesic_boundary_points_test_data(self):
        return self.generate_random_data()

    def geodesic_bvp_reverse_test_data(self):
        return self.generate_random_data_with_time()


class PointSetMetricWithArrayTestData(
    DistMixinsTestData,
    GeodesicBVPMixinsTestData,
    TestData,
):
    pass
