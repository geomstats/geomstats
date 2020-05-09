"""Unit tests for Expectation Maximization."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.expectation_maximization import RiemannianEM

TOLERANCE = 1e-3


class TestEM(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)
        dim = 2
        n_samples = 10
        self.dimension = 2
        self.space = PoincareBall(dim=self.dimension)
        self.metric = self.space.metric

        cluster_1 = gs.random.uniform(low=0.2, high=0.6, size=(n_samples, dim))
        cluster_2 = gs.random.uniform(low=-0.2, high=-0.6, size=(n_samples, dim))
        cluster_3 = gs.random.uniform(low=0, high=-0.3, size=(n_samples, dim))
        cluster_3[:, 0] = -cluster_3[:, 0]

        self.n_gaussian = 3
        self.data = gs.concatenate((cluster_1, cluster_2, cluster_3), axis=0)

    @geomstats.tests.np_only
    def test_fit(self):

        gmm_learning = RiemannianEM(
            riemannian_metric=self.metric, n_gaussian=self.n_gaussian)
        means, variances, coefficients = gmm_learning.fit(self.data)


        result = self.metric.dist(center, mean.estimate_)
        expected = 0.
        self.assertAllClose(expected, result, atol=TOLERANCE)

    @geomstats.tests.np_only
    def test_predict(self):
        X = self.data
        clustering = OnlineKMeans(
            metric=self.metric, n_clusters=3, n_repetitions=1)
        clustering.fit(X)

        point = self.data[0, :]
        prediction = clustering.predict(point)

        result = prediction
        expected = clustering.labels_[0]
        self.assertAllClose(expected, result)
