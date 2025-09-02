import geomstats.backend as gs
from geomstats.test.data import TestData

from ._base import BaseEstimatorTestData


class RobustMestimatorSOCoincideTestData(BaseEstimatorTestData):
    """Test SO matrix/vector coincidence data"""

    def estimate_coincide_test_data(self):
        """Test SO matrix/vector coincidence data"""
        return self.generate_random_data()


class HuberMeanExtremeCTestData(BaseEstimatorTestData):
    """Test huber limiting data"""

    def huber_extreme_c_test_data(self):
        """Test huber limiting data"""
        return self.generate_random_data()


class AutoGradientDescentTestData(TestData):
    """Test autograd quality data"""

    def auto_grad_descent_same_as_explicit_grad_descent_test_data(self):
        """Test autograd quality data"""
        return self.generate_random_data()
    

class VarianceTestData(BaseEstimatorTestData):
    """Test Variance quality data"""

    def variance_repeated_is_zero_test_data(self):
        """Test Variance 0 quality data"""
        return self.generate_random_data()


class VarianceEuclideanTestData(TestData):
    """Test Euclidean Variance quality data."""

    def variance_test_data(self):
        """Test Euclidean Variance quality data"""
        data = [
            dict(
                points=gs.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]),
                base_point=gs.zeros(2),
                weights=gs.array([1.0, 2.0, 1.0, 2.0]),
                expected=gs.array((1 * 5.0 + 2 * 13.0 + 1 * 25.0 + 2 * 41.0) / 6.0),
            )
        ]
        return self.generate_tests(data)


class DiffStartingPointSameResultTestData(BaseEstimatorTestData):
    """Test starting point invariance data."""

    def diff_starting_point_same_result_test_data(self):
        """Test starting point invariance data"""
        return self.generate_random_data()
    

class AutoGradientNotImplementedOnNumpyBackendTestData(BaseEstimatorTestData):
    """Test autograd not working on numpy data"""

    def auto_gradient_not_implemented_on_numpy_backend_test_data(self):
        """Test autograd not working on numpy data"""
        return self.generate_tests([{}])
      

class SameMestimatorFunctionGivenByCustomAndExplicitTestData(BaseEstimatorTestData):
    """Test custom function working data"""

    def same_m_estimator_function_given_by_custom_and_explicit_test_data(self):
        """Test custom function working data"""
        return self.generate_random_data()