import random

from geomstats.test.data import TestData


class LogNormalTestData(TestData):
    MIN_RANDOM = 2
    MAX_RANDOM = 5

    def generate_random_data(self, marks=()):
        return self.generate_tests(
            [dict(n_samples=random.randint(self.MIN_RANDOM, self.MAX_RANDOM))],
            marks=marks,
        )

    def sample_belongs_test_data(self):
        return self.generate_random_data()
