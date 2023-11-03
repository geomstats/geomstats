from ._base import BaseEstimatorTestData, ClusterMixinsTestData


class RiemannianKMedoidsTestData(ClusterMixinsTestData, BaseEstimatorTestData):
    MIN_RANDOM = 5
    MAX_RANDOM = 10

    xfails = ("n_repeated_clusters",)
