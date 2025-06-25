import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.test_cases.learning._base import BaseEstimatorTestData


class PrincipalNestedSpheresTestData(BaseEstimatorTestData):
    """
    Test data for PrincipalNestedSpheres: provides space args, sample sizes,
    and random data for each parametrized test method.
    """

    # Range for random sample sizes
    MIN_RANDOM = 5
    MAX_RANDOM = 50

    def __init__(self):
        # Test on S^2 and S^3
        self.space_args = [(2,), (3,)]
        # Tolerances for numerical comparisons
        self.atol = 1e-6

    def data_generator(self, space_args):
        sphere = Hypersphere(*space_args)

        class DataGenerator:
            def random_point(self, n_samples):
                return sphere.random_uniform(n_samples)

        return DataGenerator()

    def fit_transform_consistency_test_data(self):
        """Data for test_fit_transform_consistency"""
        return self.generate_random_data()

    def output_on_circle_test_data(self):
        """Data for test_output_on_circle"""
        return self.generate_random_data()

    def nested_levels_test_data(self):
        """Data for test_nested_levels"""
        return self.generate_random_data()
