import geomstats.backend as gs
from geomstats.test.test_case import TestCase


class AgglomerativeHierarchicalClusteringTestCase(TestCase):
    def test_fit_two_clusters(self, estimator, dataset, expected):
        if gs.unique(expected).shape[0] > 2:
            raise ValueError("Expected only two clusters")

        estimator.fit(dataset)

        clustering_labels = estimator.labels_
        result = (clustering_labels == expected).all() or (
            clustering_labels == gs.where(expected, 0, 1)
        ).all()
        expected = True
        self.assertAllClose(expected, result)
