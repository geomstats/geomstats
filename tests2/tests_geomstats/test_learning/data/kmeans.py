from ._base import BaseEstimatorTestData, ClusterMixinsTestData


class ClusterInitializationTestData(BaseEstimatorTestData):
    MIN_RANDOM = 5
    MAX_RANDOM = 10

    def initialization_belongs_test_data(self):
        return self.generate_random_data()


class RiemannianKMeansTestData(ClusterMixinsTestData, BaseEstimatorTestData):
    MIN_RANDOM = 5
    MAX_RANDOM = 10

    tolerances = {"n_repeated_clusters": {"atol": 1e-6}}


class AgainstFrechetMeanTestData(BaseEstimatorTestData):
    tolerances = {"against_frechet_mean": {"atol": 1e-1}}

    def against_frechet_mean_test_data(self):
        return self.generate_random_data()
