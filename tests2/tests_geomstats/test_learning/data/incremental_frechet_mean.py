from geomstats.test.data import TestData

from ._base import BaseEstimatorTestData, MeanEstimatorMixinsTestData


class IncrementalFrechetMeanTestData(
    MeanEstimatorMixinsTestData, BaseEstimatorTestData
):
    pass


class IncrementalFrechetMeanEuclideanTestData(TestData):
    def fit_eye_test_data(self):
        return self.generate_tests([{}])
