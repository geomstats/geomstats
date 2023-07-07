"""Unit tests for Online k-means."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.online_kmeans import OnlineKMeans


@tests.conftest.np_and_autograd_only
class TestOnlineKmeans(tests.conftest.TestCase):
    def setup_method(self):
        gs.random.seed(1234)

        self.dimension = 2
        self.space = Hypersphere(dim=self.dimension)
        self.data = self.space.random_von_mises_fisher(kappa=100, n_samples=50)

    def test_fit(self):
        X = self.data
        clustering = OnlineKMeans(
            self.space, n_clusters=1, n_repetitions=10, max_iter=50000
        )
        clustering.fit(X)

        center = clustering.cluster_centers_
        mean = FrechetMean(self.space)
        mean.fit(X)

        result = self.space.metric.dist(center, mean.estimate_)
        expected = 0.0
        self.assertAllClose(expected, result, atol=1e-3)

    def test_predict(self):
        X = self.data
        clustering = OnlineKMeans(self.space, n_clusters=3, n_repetitions=1)
        clustering.fit(X)

        point = self.data[0, :]
        prediction = clustering.predict(point)

        result = prediction
        expected = clustering.labels_[0]
        self.assertAllClose(expected, result)
