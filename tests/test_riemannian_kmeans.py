"""Unit tests for Riemannian KMeans."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry import hypersphere
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.kmeans import RiemannianKMeans


class TestRiemannianKMeansMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    @geomstats.tests.np_and_pytorch_only
    def test_hypersphere_kmeans_fit(self):
        gs.random.seed(55)

        manifold = hypersphere.Hypersphere(2)
        metric = hypersphere.HypersphereMetric(2)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        kmeans = RiemannianKMeans(metric, 1, tol=1e-3)
        kmeans.fit(x)
        center = kmeans.centroids

        mean = FrechetMean(metric=metric)
        mean.fit(x)

        result = metric.dist(center, mean.estimate_)
        expected = 0.
        self.assertAllClose(expected, result, atol=1e-2)

    @geomstats.tests.np_and_pytorch_only
    def test_hypersphere_kmeans_predict(self):
        gs.random.seed(1234)

        manifold = hypersphere.Hypersphere(2)
        metric = hypersphere.HypersphereMetric(2)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        kmeans = RiemannianKMeans(metric, 5, tol=1e-5)
        kmeans.fit(x, max_iter=100)
        result = kmeans.predict(x)

        centroids = kmeans.centroids
        expected = gs.array([int(metric.closest_neighbor_index(x_i, centroids))
                             for x_i in x])
        self.assertAllClose(expected, result)
