from geomstats.learning.aac import _AACGGPCA, _AACFrechetMean, _AACRegression
from geomstats.test.data import TestData
from geomstats.test.test_case import np_backend

from ._base import BaseEstimatorTestData, MeanEstimatorMixinsTestData

IS_NOT_NP = not np_backend()


class AACTestData(TestData):
    def init_test_data(self):
        data = [
            dict(
                estimate="frechet_mean",
                expected_type=_AACFrechetMean,
            ),
            dict(estimate="ggpca", expected_type=_AACGGPCA),
            dict(
                estimate="regression",
                expected_type=_AACRegression,
            ),
        ]

        return self.generate_tests(data)


class AACFrechetMeanTestData(MeanEstimatorMixinsTestData, BaseEstimatorTestData):
    pass


class AACGGPCATestData(BaseEstimatorTestData):
    trials = 5

    tolerances = {
        "fit_geodesic_points": {"atol": 1e-4},
    }

    def fit_geodesic_points_test_data(self):
        return self.generate_random_data()


class AACRegressionTestData(BaseEstimatorTestData):
    def fit_and_predict_constant_test_data(self):
        return self.generate_random_data()
