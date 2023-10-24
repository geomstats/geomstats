from ._base import BaseEstimatorTestData, ClusterMixinsTestData


class OnlineKMeansTestData(ClusterMixinsTestData, BaseEstimatorTestData):
    MIN_RANDOM = 5
    MAX_RANDOM = 10

    tolerances = {"n_repeated_clusters": {"atol": 1e-6}}
    xfails = ("n_repeated_clusters",)
