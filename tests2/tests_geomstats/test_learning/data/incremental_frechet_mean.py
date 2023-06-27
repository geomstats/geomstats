import random

from geomstats.test.data import TestData


class IncrementalFrechetMeanTestData(TestData):
    def estimate_belongs_test_data(self):
        return self.generate_tests([dict(n_points=random.randint(2, 10))])


class IncrementalFrechetMeanEuclideanTestData(TestData):
    def fit_eye_test_data(self):
        return self.generate_tests([{}])
