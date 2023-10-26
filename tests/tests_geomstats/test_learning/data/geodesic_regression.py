from ._base import BaseEstimatorTestData


class GeodesicRegressionTestData(BaseEstimatorTestData):
    fail_for_autodiff_exceptions = False

    tolerances = {"predict_and_score": {"atol": 0.1}}
    xfails = ("predict_and_score",)

    def loss_at_true_is_zero_test_data(self):
        return self.generate_random_data()

    def param_belongs_and_is_tangent_test_data(self):
        return self.generate_random_data()

    def predict_and_score_test_data(self):
        return self.generate_random_data()
