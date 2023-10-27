import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.agglomerative_hierarchical_clustering import (
    AgglomerativeHierarchicalClustering,
)
from geomstats.test.data import TestData


class AgglomerativeHierarchicalClusteringTestData(TestData):
    def fit_two_clusters_test_data(self):
        euclidean = Euclidean(dim=2)
        sphere = Hypersphere(dim=2)

        data = [
            dict(
                dataset=gs.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]),
                estimator=AgglomerativeHierarchicalClustering(
                    euclidean,
                    n_clusters=2,
                ),
                expected=gs.array([1, 1, 1, 0, 0, 0]),
            ),
            dict(
                dataset=gs.array(
                    [
                        [1, 0, 0],
                        [3 ** (1 / 2) / 2, 1 / 2, 0],
                        [3 ** (1 / 2) / 2, -1 / 2, 0],
                        [0, 0, 1],
                        [0, 1 / 2, 3 ** (1 / 2) / 2],
                        [0, -1 / 2, 3 ** (1 / 2) / 2],
                    ]
                ),
                estimator=AgglomerativeHierarchicalClustering(
                    sphere,
                    n_clusters=2,
                ),
                expected=gs.array([1, 1, 1, 0, 0, 0]),
            ),
        ]
        return self.generate_tests(data)
