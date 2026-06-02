from ._base import BaseEstimatorTestData


class LinearRegressionTestData(BaseEstimatorTestData):
    trials = 1

    def fit_runs_test_data(self):
        return self.generate_random_data()

    def predict_runs_test_data(self):
        return self.generate_random_data()

    def score_runs_test_data(self):
        return self.generate_random_data()
