from ._base import BaseEstimatorTestData, MeanEstimatorMixinsTestData


class ExponentialBarycenterTestData(MeanEstimatorMixinsTestData, BaseEstimatorTestData):
    pass


class AgainstFrechetMeanTestData(BaseEstimatorTestData):
    tolerances = {"against_frechet_mean": {"atol": 1e-3}}

    def against_frechet_mean_test_data(self):
        return self.generate_random_data()


class AgainstLinearMeanTestData(BaseEstimatorTestData):
    def against_linear_mean_test_data(self):
        return self.generate_random_data()
