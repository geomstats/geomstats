"""Unit tests for Riemannian KMeans."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry import hypersphere
from geomstats.geometry import spd_matrices
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.kmeans import RiemannianKMeans


class TestRiemannianKMeans(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    @geomstats.tests.np_and_pytorch_only
    def test_hypersphere_kmeans_fit(self):
        gs.random.seed(55)

        manifold = hypersphere.Hypersphere(2)
        metric = hypersphere.HypersphereMetric(2)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        kmeans = RiemannianKMeans(metric, 1, tol=1e-3, lr=1.)
        kmeans.fit(x)
        center = kmeans.centroids

        mean = FrechetMean(metric=metric, lr=1.)
        mean.fit(x)

        result = metric.dist(center, mean.estimate_)
        expected = 0.
        self.assertAllClose(expected, result)

    @geomstats.tests.np_only
    def test_spd_kmeans_fit(self):
        gs.random.seed(0)
        dim = 3
        n_points = 2
        space = spd_matrices.SPDMatrices(dim)
        data = space.random_point(n_samples=n_points)
        metric = spd_matrices.SPDMetricAffine(dim)

        kmeans = RiemannianKMeans(
            metric, n_clusters=1, lr=1.)
        kmeans.fit(data)
        result = kmeans.centroids

        mean = FrechetMean(metric=metric, point_type='matrix', max_iter=100)
        mean.fit(data)
        expected = mean.estimate_
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_hypersphere_kmeans_predict(self):
        gs.random.seed(1234)
        dim = 2

        manifold = hypersphere.Hypersphere(dim)
        metric = hypersphere.HypersphereMetric(dim)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        kmeans = RiemannianKMeans(metric, 5, tol=1e-5, lr=1.)
        kmeans.fit(x)
        result = kmeans.predict(x)

        centroids = kmeans.centroids
        expected = gs.array(
            [int(metric.closest_neighbor_index(x_i, centroids))
             for x_i in x])
        self.assertAllClose(expected, result)
