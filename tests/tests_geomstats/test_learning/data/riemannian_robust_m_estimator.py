import geomstats.backend as gs
from geomstats.test.data import TestData

from ._base import BaseEstimatorTestData


class RobustMestimatorSOCoincideTestData(BaseEstimatorTestData):
    """Test SO matrix/vector coincidence data"""

    def estimate_coincide_test_data(self):
        """Test SO matrix/vector coincidence data"""
        return self.generate_random_data()


class LimitingCofHuberLossTestData(BaseEstimatorTestData):
    """Test huber limiting data"""

    def limiting_c_huber_loss_test_data(self):
        """Test huber limiting data"""
        return self.generate_random_data()


class AutoGradientDescentOneStepTestData(TestData):
    """Test autograd quality data"""

    def onestep_auto_grad_descent_same_as_explicit_grad_descent_test_data(self):
        """Test autograd quality data"""
        return self.generate_random_data()
    

class AutoGradientDescentResultTestData(TestData):
    """Test autograd quality data"""

    def auto_grad_descent_result_same_as_explicit_grad_descent_test_data(self):
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
            ),
            dict(
                points=gs.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]),
                base_point=None,
                weights=gs.array([1.0, 1.0, 1.0, 1.0, 1.0]),
                expected=gs.array((1 * 8.0 + 1 * 2.0 + 1 * 0.0 + 1 * 2.0 + 1 * 8.0) / 5.0),
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

    def numpy_backend_autograd_error_test_data(self):
        """Test autograd not working on numpy data"""
        return self.generate_tests([{}])

    def custom_m_estimator_loss_provided_test_data(self):
        return self.generate_tests([{}])

    def custom_m_estimator_loss_not_provided_test_data(self):
        return self.generate_tests([{}])


class NotImplementedBlockingsTestData(BaseEstimatorTestData):
    """Test autograd not working on numpy data"""

    def hampel_loss_with_fault_critical_value_test_data(self):
        """Test autograd not working on numpy data"""
        return self.generate_tests([{}])

    def invalid_m_estimator_test_data(self):
        return self.generate_tests([{}])

    def one_point_fit_test_data(self):
        return self.generate_tests([{}])

    def one_point_fit_d_test_data(self):
        return self.generate_tests([{}])


class SameMestimatorFunctionGivenByCustomAndExplicitTestData(BaseEstimatorTestData):
    """Test custom function working data"""

    def same_m_estimator_function_given_by_custom_and_explicit_test_data(self):
        """Test custom function working data"""
        return self.generate_random_data()


class MestimatorCustomFunctionDifferentInputArgsTestData(BaseEstimatorTestData):
    """Test custom function input change data"""
    
    def custom_function_different_input_arguments_test_data(self):
        """Test custom function input change data"""
        return self.generate_random_data()
