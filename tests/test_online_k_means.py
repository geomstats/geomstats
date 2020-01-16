"""
Unit tests for Online k-means.
"""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.online_k_means import OnlineKmeans

TOLERANCE = 1e-3


class TestOnlineKmeansMethods(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.dimension = 2
        self.space = Hypersphere(dimension=self.dimension)
        self.metric = self.space.metric
        self.data = self.space.random_von_mises_fisher(
            kappa=100, n_samples=50)

    @geomstats.tests.np_only
    def test_fit(self):
        X = self.data
        clustering = OnlineKmeans(metric=self.metric, n_clusters=1,
                                  n_repetitions=10)
        clustering.fit(X)

        center = clustering.cluster_centers_
        mean = self.metric.mean(X)
        result = self.metric.dist(center, mean)
        expected = 0.
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_predict(self):
        X = self.data
        clustering = OnlineKmeans(metric=self.metric, n_clusters=3,
                                  n_repetitions=1)
        clustering.fit(X)

        point = self.data[0, :]
        prediction = clustering.predict(point)

        result = prediction
        expected = clustering.labels_[0]
        self.assertAllClose(expected, result)
