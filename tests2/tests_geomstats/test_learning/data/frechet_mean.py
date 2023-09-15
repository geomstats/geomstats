import random

from geomstats.test.data import TestData

from ._base import BaseEstimatorTestData, MeanEstimatorMixinsTestData


class FrechetMeanTestData(MeanEstimatorMixinsTestData, BaseEstimatorTestData):
    fail_for_autodiff_exceptions = False

    def logs_at_mean_test_data(self):
        return self.generate_tests([{}])

    def weighted_mean_two_points_test_data(self):
        return self.generate_tests([{}])


class CircularMeanTestData(FrechetMeanTestData):
    skips = ("weighted_mean_two_points",)

    def against_optimization_test_data(self):
        # TODO: something wrong for certain n_points?
        return self.generate_tests([dict(n_points=10, atol=1e-4)])


class BatchGradientDescentTestData(TestData):
    def against_default_test_data(self):
        return self.generate_tests(
            [dict(n_points=random.randint(2, 10), n_reps=random.randint(2, 5))]
        )
