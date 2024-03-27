from geomstats.test.data import TestData


class InterpolatorTestData(TestData):
    def interpolate_with_given_data_test_data(self):
        return self.generate_tests([dict()])

    def interpolate_half_interval_test_data(self):
        return self.generate_tests([dict()])
