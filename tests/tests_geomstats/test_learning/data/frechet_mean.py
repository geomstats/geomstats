import random

import geomstats.backend as gs
from geomstats.test.data import TestData

from ._base import BaseEstimatorTestData, MeanEstimatorMixinsTestData


class PointSetFrechetMeanTestData(MeanEstimatorMixinsTestData, BaseEstimatorTestData):
    atol = 1e-2

    def weighted_mean_two_points_test_data(self):
        return self.generate_tests([{}])


class FrechetMeanTestData(MeanEstimatorMixinsTestData, BaseEstimatorTestData):
    fail_for_autodiff_exceptions = False

    def logs_at_mean_test_data(self):
        return self.generate_tests([{}])

    def weighted_mean_two_points_test_data(self):
        return self.generate_tests([{}])


class FrechetMeanSOCoincideTestData(BaseEstimatorTestData):
    def estimate_coincide_test_data(self):
        return self.generate_random_data()


class CircularMeanTestData(FrechetMeanTestData):
    skips = ("weighted_mean_two_points",)

    def against_optimization_test_data(self):
        # TODO: something wrong for certain n_points?
        return self.generate_tests([dict(n_points=10, atol=1e-4)])


class LinearMeanEuclideaTestData(TestData):
    def fit_test_data(self):
        data = [
            dict(
                X=gs.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]),
                weights=gs.array([1.0, 2.0, 1.0, 2.0]),
                expected=gs.array([16.0 / 6.0, 22.0 / 6.0]),
            )
        ]

        return self.generate_tests(data)


class VarianceTestData(BaseEstimatorTestData):
    def variance_repeated_is_zero_test_data(self):
        return self.generate_random_data()


class VarianceEuclideanTestData(TestData):
    def variance_test_data(self):
        data = [
            dict(
                points=gs.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]),
                base_point=gs.zeros(2),
                weights=gs.array([1.0, 2.0, 1.0, 2.0]),
                expected=gs.array((1 * 5.0 + 2 * 13.0 + 1 * 25.0 + 2 * 41.0) / 6.0),
            )
        ]
        return self.generate_tests(data)


class BatchGradientDescentTestData(TestData):
    def against_default_test_data(self):
        return self.generate_tests(
            [dict(n_points=random.randint(2, 10), n_reps=random.randint(2, 5))]
        )
