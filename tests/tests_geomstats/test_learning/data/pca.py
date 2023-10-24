from ._base import BaseEstimatorTestData


class TangentPCATestData(BaseEstimatorTestData):
    MIN_RANDOM = 5
    MAX_RANDOM = 10

    def fit_inverse_transform_test_data(self):
        return self.generate_random_data()

    def fit_transform_and_transform_after_fit_test_data(self):
        return self.generate_random_data()

    def n_components_test_data(self):
        return self.generate_random_data()

    def n_components_explained_variance_ratio_test_data(self):
        return self.generate_random_data()

    def n_components_mle_test_data(self):
        return self.generate_random_data()
