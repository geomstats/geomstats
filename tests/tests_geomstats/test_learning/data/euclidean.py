from ._base import BaseEstimatorTestData


class LinearRegressionTestData(BaseEstimatorTestData):
    trials = 1

    def runs_test_data(self):
        return self.generate_random_data()
