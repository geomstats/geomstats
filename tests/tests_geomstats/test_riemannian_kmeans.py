"""Unit tests for Riemannian KMeans."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry import hypersphere, spd_matrices
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.kmeans import RiemannianKMeans


@geomstats.tests.np_and_autograd_only
class TestRiemannianKMeans(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def test_hypersphere_kmeans_fit(self):
        gs.random.seed(55)

        manifold = hypersphere.Hypersphere(2)
        metric = hypersphere.HypersphereMetric(2)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        kmeans = RiemannianKMeans(metric, 1, init_step_size=1.0, tol=1e-3)
        kmeans.fit(x)
        center = kmeans.centroids

        mean = FrechetMean(metric=metric, init_step_size=1.0)
        mean.fit(x)

        result = metric.dist(center, mean.estimate_)
        expected = 0.0
        self.assertAllClose(expected, result)

    def test_spd_kmeans_fit(self):
        gs.random.seed(0)
        dim = 3
        n_points = 2
        space = spd_matrices.SPDMatrices(dim)
        data = space.random_point(n_samples=n_points)
        metric = spd_matrices.SPDMetricAffine(dim)

        kmeans = RiemannianKMeans(metric, n_clusters=1, init_step_size=1.0)
        kmeans.fit(data)
        result = kmeans.centroids

        mean = FrechetMean(metric=metric, max_iter=100, point_type="matrix")
        mean.fit(data)
        expected = mean.estimate_
        self.assertAllClose(result, expected)

    def test_hypersphere_kmeans_predict(self):
        gs.random.seed(1234)
        dim = 2

        manifold = hypersphere.Hypersphere(dim)
        metric = hypersphere.HypersphereMetric(dim)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        kmeans = RiemannianKMeans(metric, 5, init_step_size=1.0, tol=1e-5)
        kmeans.fit(x)
        result = kmeans.predict(x)

        centroids = kmeans.centroids
        expected = gs.array(
            [int(metric.closest_neighbor_index(x_i, centroids)) for x_i in x]
        )
        self.assertAllClose(expected, result)

    def test_hypersphere_kmeans_init_kmeanspp(self):
        gs.random.seed(55)

        manifold = hypersphere.Hypersphere(2)
        metric = hypersphere.HypersphereMetric(2)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=200)

        n_clusters = 3
        kmeans = RiemannianKMeans(
            metric, n_clusters, init_step_size=1.0, tol=1e-3, init="kmeans++"
        )
        kmeans.fit(x)
        centroids = kmeans.centroids
        result = centroids.shape
        expected = (n_clusters, 3)
        self.assertAllClose(expected, result)

    def test_hypersphere_kmeans_init_array(self):
        gs.random.seed(55)

        n_features = 3
        n_clusters = 3

        manifold = hypersphere.Hypersphere(n_features - 1)

        x = manifold.random_von_mises_fisher(kappa=100, n_samples=200)
        y = manifold.random_von_mises_fisher(kappa=10, n_samples=n_clusters)

        kmeans = RiemannianKMeans(
            manifold.metric, n_clusters, init_step_size=1.0, tol=1e-3, init=y
        )
        kmeans.fit(x)
        centroids = kmeans.centroids
        result = centroids.shape
        expected = (n_clusters, n_features)
        self.assertAllClose(expected, result)
