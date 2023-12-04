from geomstats.test.data import TestData

from .point_set import PointSetMetricWithArrayTestData


class AlignerTestData(TestData):
    def align_vec_test_data(self):
        return self.generate_vec_data()


class AlignerCmpTestData(TestData):
    def align_test_data(self):
        return self.generate_random_data()


class QuotientMetricWithArrayTestData(PointSetMetricWithArrayTestData):
    pass
