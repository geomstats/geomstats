import random

from geomstats.test.data import TestData


class FrechetMeanTestData(TestData):
    def one_point_test_data(self):
        return self.generate_tests([{}])

    def n_times_same_point_test_data(self):
        return self.generate_tests([dict(n_reps=random.randint(2, 5))])

    def estimate_belongs_test_data(self):
        return self.generate_tests([dict(n_points=random.randint(2, 10))])

    def logs_at_mean_test_data(self):
        return self.generate_tests([{}])


class ElasticMeanTestData(FrechetMeanTestData):
    pass


class CircularMeanTestData(FrechetMeanTestData):
    def against_optimization_test_data(self):
        # TODO: something wrong for certain n_points?
        return self.generate_tests([dict(n_points=10, atol=1e-4)])


class BatchGradientDescentTestData(TestData):
    def against_default_test_data(self):
        return self.generate_tests(
            [dict(n_points=random.randint(2, 10), n_reps=random.randint(2, 5))]
        )
