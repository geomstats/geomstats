import geomstats.backend as gs
from geomstats.test.test_case import TestCase


class AgglomerativeHierarchicalClusteringTestCase(TestCase):
    def test_fit_two_clusters(self, estimator, dataset, expected):
        if gs.unique(expected).shape[0] > 2:
            raise ValueError("Expected only two clusters")

        estimator.fit(dataset)

        clustering_labels = estimator.labels_
        self.assertAllEqual(clustering_labels, expected)
