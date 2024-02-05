from ._base import BaseEstimatorTestData


class WrappedGaussianProcessTestData(BaseEstimatorTestData):
    MIN_RANDOM = 10
    MAX_RANDOM = 20

    tolerances = {"predict_at_train_zero_std": {"atol": 1e-4}}

    def score_at_train_is_one_test_data(self):
        return self.generate_random_data()

    def predict_at_train_belongs_test_data(self):
        return self.generate_random_data()

    def predict_at_train_zero_std_test_data(self):
        return self.generate_random_data()

    def sample_y_at_train_belongs_test_data(self):
        return self.generate_random_data()
