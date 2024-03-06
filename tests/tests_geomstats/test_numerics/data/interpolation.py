from geomstats.test.data import TestData


class UniformUnitIntervalLinearInterpolatorTestData(TestData):
    def interpolate_uniformly_test_data(self):
        return self.generate_tests([dict()])
