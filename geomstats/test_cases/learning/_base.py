from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase

# TODO: estimate_belongs as a first general test?


class BaseEstimatorTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.estimator.space)
