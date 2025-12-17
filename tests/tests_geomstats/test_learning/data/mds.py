from geomstats.test.data import TestData

from ._base import BaseEstimatorTestData


class MetricMDSTestData(BaseEstimatorTestData):
    pass


class MetricMDSEuclideanTestData(TestData):
    def fit_eye_test_data(self):
        print(self.generate_tests([{}]))
        return self.generate_tests([{}])
