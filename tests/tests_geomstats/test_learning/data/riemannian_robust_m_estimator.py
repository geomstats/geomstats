import geomstats.backend as gs
from geomstats.test.data import TestData

from ._base import BaseEstimatorTestData


class RobustMestimatorSOCoincideTestData(BaseEstimatorTestData):
    def estimate_coincide_test_data(self):
        return self.generate_random_data()


class HuberMeanExtremeCTestData(BaseEstimatorTestData):
    def huber_extreme_c_test_data(self):
        return self.generate_random_data()


class AutoGradientDescentTestData(TestData):
    def auto_grad_descent_same_as_explicit_grad_descent_test_data(self):
        return self.generate_random_data()
    

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
